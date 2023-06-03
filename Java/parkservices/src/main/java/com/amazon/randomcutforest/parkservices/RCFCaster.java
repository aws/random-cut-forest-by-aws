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
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_OUTPUT_AFTER_FRACTION;
import static java.lang.Math.abs;
import static java.lang.Math.max;

import java.util.Optional;
import java.util.function.BiFunction;

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

    public static boolean USE_INTERPOLATION_IN_DISTRIBUTION = true;

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
            if (outputAfter.isPresent()) {
                startNormalization = Optional.of(outputAfter.get() + shingleSize - 1);
            } else {
                startNormalization = Optional.of((int) (sampleSize * DEFAULT_OUTPUT_AFTER_FRACTION) + shingleSize - 1);
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
        checkArgument(errorHorizon >= 2 * forecastHorizon,
                "Error (used to compute interval precision of forecasts) horizon should be at least twice as large as forecast horizon");
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
        ForecastDescriptor initial = new ForecastDescriptor(inputPoint, timestamp, forecastHorizon);
        ForecastDescriptor answer;
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        try {
            if (cacheDisabled) {
                // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            // forecast first
            // if calibration is not set then RCF has to produce the model errors; otherwise
            // RCF can focus on p50
            double centralty = (calibrationMethod == Calibration.NONE) ? 1.0 - 2 * errorHandler.percentile : 1.0;
            TimedRangeVector timedForecast = extrapolate(forecastHorizon, true, centralty);
            // anomaly computation next; and subsequent update
            answer = preprocessor.postProcess(
                    predictorCorrector.detect(preprocessor.preProcess(initial, lastAnomalyDescriptor, forest),
                            lastAnomalyDescriptor, forest),
                    lastAnomalyDescriptor, forest);
            answer.setTimedForecast(timedForecast);

            if (answer.internalTimeStamp >= forest.getShingleSize() - 1 + forest.getOutputAfter()) {
                errorHandler.update(answer, calibrationMethod);
            }
        } finally {
            if (cacheDisabled) {
                // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }

        if (answer.getAnomalyGrade() > 0) {
            lastAnomalyDescriptor = answer.copyOf();
        }
        return answer;
    }

    public RangeVector computeErrorPercentile(double percentile, BiFunction<Float, Float, Float> error) {
        return computeErrorPercentile(percentile, errorHorizon, error);
    }

    public RangeVector computeErrorPercentile(double percentile, int newHorizon,
            BiFunction<Float, Float, Float> error) {
        return errorHandler.computeErrorPercentile(percentile, newHorizon, error);
    }
}
