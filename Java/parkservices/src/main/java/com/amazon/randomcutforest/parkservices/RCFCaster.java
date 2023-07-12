/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.abs;
import static java.lang.Math.max;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class RCFCaster extends ThresholdedRandomCutForest {

    public static double DEFAULT_ERROR_PERCENTILE = 0.1;

    public static Calibration DEFAULT_CALIBRATION = Calibration.SIMPLE;

    public static BiFunction<Float, Float, Float> defaultError = (x, y) -> x - y;

    public static BiFunction<Float, Float, Float> alternateError = (x, y) -> 2 * (x - y) / (abs(x) + abs(y));

    protected int forecastHorizon;
    protected ErrorHandler errorHandler;
    protected int errorHorizon;
    protected Calibration calibrationMethod;

    public static class Builder extends ThresholdedRandomCutForest.Builder<Builder> {
        int forecastHorizon;
        int errorHorizon;
        double percentile = DEFAULT_ERROR_PERCENTILE;
        protected Calibration calibrationMethod = DEFAULT_CALIBRATION;

        Builder() {
            super();
            // changing the default;
            transformMethod = TransformMethod.NORMALIZE;
        }

        public Builder forecastHorizon(int horizon) {
            this.forecastHorizon = horizon;
            return this;
        }

        public Builder errorHorizon(int errorHorizon) {
            this.errorHorizon = errorHorizon;
            return this;
        }

        public Builder percentile(double percentile) {
            this.percentile = percentile;
            return this;
        }

        public Builder calibration(Calibration calibrationMethod) {
            this.calibrationMethod = calibrationMethod;
            return this;
        }

        @Override
        public RCFCaster build() {
            checkArgument(forecastHorizon > 0, "need non-negative horizon");
            checkArgument(shingleSize > 0, "need shingle size > 1");
            checkArgument(forestMode != ForestMode.STREAMING_IMPUTE,
                    "error estimation with on the fly imputation should not be abstracted, "
                            + "either estimate errors outside of this object "
                            + "or perform on the fly imputation outside this code");
            checkArgument(forestMode != ForestMode.TIME_AUGMENTED,
                    "error estimation when time is used as a field in the forest should not be abstracted"
                            + "perform estimation outside this code");
            checkArgument(!internalShinglingEnabled.isPresent() || internalShinglingEnabled.get(),
                    "internal shingling only");
            if (errorHorizon == 0) {
                errorHorizon = max(sampleSize, 2 * forecastHorizon);
            }
            validate();
            return new RCFCaster(this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public RCFCaster(Builder builder) {
        super(builder);
        forecastHorizon = builder.forecastHorizon;
        errorHorizon = builder.errorHorizon;
        errorHandler = new ErrorHandler(builder);
        calibrationMethod = builder.calibrationMethod;
    }

    // for mappers
    public RCFCaster(ForestMode forestMode, TransformMethod transformMethod, ScoringStrategy scoringStrategy,
            RandomCutForest forest, PredictorCorrector predictorCorrector, Preprocessor preprocessor,
            RCFComputeDescriptor descriptor, int forecastHorizon, ErrorHandler errorHandler, int errorHorizon,
            Calibration calibrationMethod) {
        super(forestMode, transformMethod, scoringStrategy, forest, predictorCorrector, preprocessor, descriptor);
        this.forecastHorizon = forecastHorizon;
        this.errorHandler = errorHandler;
        this.errorHorizon = errorHorizon;
        this.calibrationMethod = calibrationMethod;
    }

    /**
     * a single call that preprocesses data, compute score/grade, generates forecast
     * and updates state
     *
     * @param inputPoint current input point
     * @param timestamp  time stamp of input
     * @return forecast descriptor for the current input point
     */

    @Override
    public ForecastDescriptor process(double[] inputPoint, long timestamp) {
        return process(inputPoint, timestamp, null);
    }

    void augmentForecast(ForecastDescriptor answer) {
        answer.setScoringStrategy(scoringStrategy);
        answer = preprocessor.postProcess(predictorCorrector
                .detect(preprocessor.preProcess(answer, lastAnomalyDescriptor, forest), lastAnomalyDescriptor, forest),
                lastAnomalyDescriptor, forest);
        if (saveDescriptor(answer)) {
            lastAnomalyDescriptor = answer.copyOf();
        }
        TimedRangeVector timedForecast = new TimedRangeVector(
                forest.getDimensions() * forecastHorizon / preprocessor.getShingleSize(), forecastHorizon);

        // note that internal timestamp of answer is 1 step in the past
        // outputReady corresponds to first (and subsequent) forecast
        if (forest.isOutputReady()) {
            errorHandler.updateActuals(answer.getCurrentInput(), answer.getPostDeviations());
            errorHandler.augmentDescriptor(answer);

            // if the last point was an anomaly then it would be corrected
            // note that forecast would show up for a point even when the anomaly score is 0
            // because anomaly designation needs X previous points and forecast needs X
            // points; note that calibration would have been performed already
            timedForecast = extrapolate(forecastHorizon);

            // note that internal timestamp of answer is 1 step in the past
            // outputReady corresponds to first (and subsequent) forecast

            errorHandler.updateForecasts(timedForecast.rangeVector);
        }
        answer.setTimedForecast(timedForecast);
    }

    /**
     * a single call that preprocesses data, compute score/grade and updates state
     * when the current input has potentially missing values
     *
     * @param inputPoint    current input point
     * @param timestamp     time stamp of input
     * @param missingValues this is not meaningful for forecast; but kept as a
     *                      parameter since it conforms to (sometimes used)
     *                      ThresholdedRCF
     * @return forecast descriptor for the current input point
     */

    @Override
    public ForecastDescriptor process(double[] inputPoint, long timestamp, int[] missingValues) {
        checkArgument(missingValues == null, "on the fly imputation and error estimation should not mix");
        ForecastDescriptor answer = new ForecastDescriptor(inputPoint, timestamp, forecastHorizon);
        answer.setScoringStrategy(scoringStrategy);
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        try {
            if (cacheDisabled) {
                // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            augmentForecast(answer);
        } finally {
            if (cacheDisabled) {
                // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }

        return answer;
    }

    public void calibrate(Calibration calibration, RangeVector ranges) {
        errorHandler.calibrate(calibration, ranges);
    }

    @Override
    public TimedRangeVector extrapolate(int horizon, boolean correct, double centrality) {
        return this.extrapolate(calibrationMethod, horizon, correct, centrality);
    }

    public TimedRangeVector extrapolate(Calibration calibration, int horizon, boolean correct, double centrality) {
        TimedRangeVector answer = super.extrapolate(horizon, correct, centrality);
        calibrate(calibration, answer.rangeVector);
        return answer;
    }

    @Override
    public List<AnomalyDescriptor> processSequentially(double[][] data, Function<AnomalyDescriptor, Boolean> filter) {
        ArrayList<AnomalyDescriptor> answer = new ArrayList<>();
        if (data != null) {
            if (data.length > 0) {
                boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
                try {
                    if (cacheDisabled) { // turn caching on temporarily
                        forest.setBoundingBoxCacheFraction(1.0);
                    }
                    long timestamp = preprocessor.getInternalTimeStamp();
                    int length = preprocessor.getInputLength();
                    for (double[] point : data) {
                        checkArgument(point.length == length, " nonuniform lengths ");
                        ForecastDescriptor description = new ForecastDescriptor(point, timestamp++, forecastHorizon);
                        augmentForecast(description);
                        if (filter.apply(description)) {
                            answer.add(description);
                        }
                    }
                } finally {
                    if (cacheDisabled) { // turn caching off
                        forest.setBoundingBoxCacheFraction(0);
                    }
                }
            }
        }
        return answer;
    }
}
