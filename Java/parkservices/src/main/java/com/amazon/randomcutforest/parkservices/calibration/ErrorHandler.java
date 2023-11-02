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

package com.amazon.randomcutforest.parkservices.calibration;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.parkservices.RCFCaster.DEFAULT_ERROR_PERCENTILE;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.Optional;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.PredictiveRandomCutForest;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;
import com.amazon.randomcutforest.parkservices.config.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.statistics.Deviation;

// we recommend the article "Regret in the On-Line Decision Problem", by Foster and Vohra,
// Games and Economic Behavior, Vol=29 (1-2), 1999
// the discussion is applicable to non-regret scenarios as well; but essentially boils down to
// fixed point/minimax computation. One could use multiplicative update type methods which would be
// uniform over all quantiles, provided sufficient data and a large enough calibration horizon.
// Multiplicative updates are scale free -- but providing scale free forecasting over a stream raises the
// issue "what is the current scale of the stream". While such questions can be answered, that discussion
// can be involved and out of current scope of this library. We simplify the issue to calibrating two
// fixed quantiles and hence additive updates are reasonable.

@Getter
@Setter
public class ErrorHandler {

    int sequenceIndex;
    double percentile;
    int forecastHorizon;
    int errorHorizon;
    // the following arrays store the state of the sequential computation
    // these can be optimized -- for example once could store the errors; which
    // would see much fewer increments.
    // However, for a small enough errorHorizon, the generality of
    // changing the error function
    // outweighs the benefit of recomputation. The search in the ensemble tree is
    // still a larger bottleneck than
    // these computations at the moment; not to mention issues of saving and
    // restoring state.
    protected RangeVector[] pastForecasts;

    RangeVector errorDistribution;
    DiVector errorRMSE;
    float[] errorMean;
    Deviation[] intervalPrecision;
    Deviation[] rmseHighDeviations;
    Deviation[] rmseLowDeviations;
    float[] lowerLimit;
    float[] upperLimit;
    double[] lastInput;
    PredictiveRandomCutForest estimator;
    float[] lastDataDeviations;

    // We keep the multipliers defined for potential
    // future use.

    RangeVector multipliers;
    RangeVector adders;

    public ErrorHandler(Builder builder) {
        checkArgument(builder.forecastHorizon > 0, "has to be positive");
        checkArgument(builder.errorHorizon >= builder.forecastHorizon,
                "intervalPrecision horizon should be at least as large as forecast horizon");
        checkArgument(builder.errorHorizon <= 1024, "reduce error horizon");
        forecastHorizon = builder.forecastHorizon;
        errorHorizon = builder.errorHorizon;
        int inputLength = (builder.dimensions / builder.shingleSize);
        int length = inputLength * forecastHorizon;
        percentile = builder.percentile;
        pastForecasts = new RangeVector[forecastHorizon];
        for (int i = 0; i < forecastHorizon; i++) {
            pastForecasts[i] = new RangeVector(length);
        }
        sequenceIndex = 0;
        lastInput = new double[inputLength];
        rmseHighDeviations = new Deviation[length];
        rmseLowDeviations = new Deviation[length];
        intervalPrecision = new Deviation[length];
        for (int i = 0; i < length; i++) {
            rmseHighDeviations[i] = new Deviation(1.0 / errorHorizon);
            rmseLowDeviations[i] = new Deviation(1.0 / errorHorizon);
            intervalPrecision[i] = new Deviation(1.0 / errorHorizon);
        }
        errorMean = new float[length];
        errorRMSE = new DiVector(length);
        lastDataDeviations = new float[inputLength];
        errorDistribution = new RangeVector(length);
        if (builder.upperLimit.isPresent()) {
            checkArgument(builder.upperLimit.get().length == inputLength, "incorrect length");
            upperLimit = Arrays.copyOf(builder.upperLimit.get(), inputLength);
        } else {
            upperLimit = new float[inputLength];
            Arrays.fill(upperLimit, Float.MAX_VALUE);
        }
        if (builder.lowerLimit.isPresent()) {
            checkArgument(builder.lowerLimit.get().length == inputLength, "incorrect length");
            for (int y = 0; y < inputLength; y++) {
                checkArgument(builder.lowerLimit.get()[y] <= upperLimit[y], "incorrect limits");
            }
            lowerLimit = Arrays.copyOf(builder.lowerLimit.get(), inputLength);
        } else {
            lowerLimit = new float[inputLength];
            Arrays.fill(lowerLimit, -Float.MAX_VALUE);
        }
        if (builder.useRCF) {
            estimator = new PredictiveRandomCutForest.Builder<>().inputDimensions(3 * lastInput.length + 1)
                    .outputAfter(100).transformMethod(TransformMethod.NORMALIZE).startNormalization(99).build();
        }
    }

    // for mappers
    public ErrorHandler(int errorHorizon, int forecastHorizon, int sequenceIndex, double percentile, int inputLength,
            float[] pastForecastsFlattened, float[] lastDataDeviations, double[] lastInput, Deviation[] deviations,
            PredictiveRandomCutForest estimator, float[] auxiliary) {
        checkArgument(forecastHorizon > 0, " incorrect forecast horizon");
        checkArgument(errorHorizon >= forecastHorizon, "incorrect error horizon");
        checkArgument(inputLength > 0, "incorrect parameters");
        checkArgument(sequenceIndex >= 0, "cannot be negative");
        checkArgument(Math.abs(percentile - 0.25) < 0.24, "has to be between (0,0.5) ");
        checkArgument(deviations.length == 3 * inputLength * forecastHorizon, "incorrect length");
        checkArgument(lastInput.length == inputLength, "incoorect length");

        this.sequenceIndex = sequenceIndex;
        this.errorHorizon = errorHorizon;
        this.percentile = percentile;
        this.forecastHorizon = forecastHorizon;
        this.pastForecasts = new RangeVector[forecastHorizon];
        this.lastInput = Arrays.copyOf(lastInput, lastInput.length);

        int length = forecastHorizon * inputLength;
        checkArgument(lastDataDeviations.length >= inputLength, "incorrect length");
        this.lastDataDeviations = Arrays.copyOf(lastDataDeviations, lastDataDeviations.length);
        this.errorMean = new float[length];
        this.errorRMSE = new DiVector(length);
        this.errorDistribution = new RangeVector(length);
        this.intervalPrecision = new Deviation[inputLength * forecastHorizon];
        this.rmseHighDeviations = new Deviation[inputLength * forecastHorizon];
        this.rmseLowDeviations = new Deviation[inputLength * forecastHorizon];
        for (int y = 0; y < inputLength * forecastHorizon; y++) {
            this.intervalPrecision[y] = deviations[y].copy();
            this.rmseHighDeviations[y] = deviations[y + inputLength * forecastHorizon].copy();
            this.rmseLowDeviations[y] = deviations[y + 2 * inputLength * forecastHorizon].copy();
        }
        lowerLimit = new float[inputLength];
        Arrays.fill(lowerLimit, -Float.MAX_VALUE);
        upperLimit = new float[inputLength];
        Arrays.fill(upperLimit, Float.MAX_VALUE);
        this.estimator = estimator;
        int arrayLength = (pastForecastsFlattened == null) ? 0 : pastForecastsFlattened.length / (3 * length);
        if (pastForecastsFlattened != null) {
            for (int i = 0; i < arrayLength; i++) {
                float[] values = Arrays.copyOfRange(pastForecastsFlattened, i * 3 * length, (i * 3 + 1) * length);
                float[] upper = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 1) * length, (i * 3 + 2) * length);
                float[] lower = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 2) * length, (i * 3 + 3) * length);
                pastForecasts[i] = new RangeVector(values, upper, lower);
            }
        }
        for (int i = arrayLength; i < forecastHorizon; i++) {
            pastForecasts[i] = new RangeVector(length);
        }
        recomputeErrors(lastInput);
    }

    public void setUpperLimit(float[] upperLimit) {
        if (upperLimit != null) {
            checkArgument(upperLimit.length == this.upperLimit.length, "incorrect Length");
            System.arraycopy(upperLimit, 0, this.upperLimit, 0, upperLimit.length);
        }
    }

    public void setLowerLimit(float[] lowerLimit) {
        if (lowerLimit != null) {
            checkArgument(lowerLimit.length == this.lowerLimit.length, "incorrect Length");
            for (int i = 0; i < lowerLimit.length; i++) {
                checkArgument(lowerLimit[i] <= this.upperLimit[i], "lower limit is higher than upper limit");
                this.lowerLimit[i] = lowerLimit[i];
            }
        }
    }

    /**
     * updates the stored information (actuals) and recomputes the calibrations
     * 
     * @param input      the actual input
     * @param deviations the deviations (post the current input)
     */

    public void updateActuals(double[] input, double[] deviations) {
        int arrayLength = pastForecasts.length;
        int inputLength = input.length;

        if (sequenceIndex > 0) {
            // sequenceIndex indicates the first empty place for input
            // note the predictions have already been stored
            int inputIndex = (sequenceIndex + arrayLength - 1) % arrayLength;
            float[] errorTuple = new float[3 * inputLength + 1];
            for (int y = 0; y < inputLength; y++) {
                errorTuple[y] = (float) input[y];
            }
            for (int j = 0; j < forecastHorizon; j++) {
                if (sequenceIndex > j) {
                    for (int i = 0; i < inputLength; i++) {
                        RangeVector a = pastForecasts[inputIndex];
                        int offset = j * inputLength;
                        errorTuple[inputLength + 1] = j;
                        if (input[i] <= a.upper[offset + i] && input[i] >= a.lower[offset + i]) {
                            intervalPrecision[offset + i].update(1.0);
                        } else {
                            intervalPrecision[offset + i].update(0);
                        }
                        double error = input[i] - a.values[offset + i];
                        if (error >= 0) {
                            rmseHighDeviations[offset + i].update(error);
                            rmseLowDeviations[offset + i].update(0);
                            errorTuple[inputLength + 1 + i] = (float) error;
                            errorTuple[2 * inputLength + 1 + i] = 0;
                        } else {
                            rmseLowDeviations[offset + i].update(error);
                            rmseHighDeviations[offset + i].update(0);
                            errorTuple[2 * inputLength + 1 + i] = (float) (error);
                            errorTuple[inputLength + 1 + i] = 0;
                        }
                    }
                    if (estimator != null) {
                        estimator.update(errorTuple, 0L);
                    }
                }
                inputIndex = (inputIndex + arrayLength - 1) % arrayLength;
            }
        }
        // sequence index is increased first so that recomputeErrors is idempotent;
        // that is, they are only state dependent and not event dependent
        ++sequenceIndex;
        lastDataDeviations = toFloatArray(deviations);
        recomputeErrors(input);
    }

    void recomputeErrors(double[] lastInput) {
        int inputLength = lastInput.length;
        double a;
        if (estimator != null) {
            a = (double) (sequenceIndex) / (estimator.getForest().getOutputAfter());
        } else {
            a = (double) (sequenceIndex) / (10 * forecastHorizon);
        }
        float[] query = new float[inputLength * 3 + 1];
        System.arraycopy(toFloatArray(lastInput), 0, query, 0, inputLength);
        float[] errorHigh = new float[intervalPrecision.length];
        float[] errorLow = new float[intervalPrecision.length];
        if (a < 1) {
            for (int y = 0; y < intervalPrecision.length; y++) {
                errorRMSE.high[y] = errorRMSE.low[y] = lastDataDeviations[y % inputLength];
                errorHigh[y] = errorLow[y] = lastDataDeviations[y % inputLength];
            }
        } else {
            if (a < 2) {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    double offset = (2 - a) * lastDataDeviations[y % inputLength];
                    errorRMSE.high[y] = (offset + (a - 1) * rmseHighDeviations[y].getDeviation());
                    errorRMSE.low[y] = (offset + (a - 1) * rmseLowDeviations[y].getDeviation());
                }
            } else {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    errorRMSE.high[y] = rmseHighDeviations[y].getDeviation();
                    errorRMSE.low[y] = rmseLowDeviations[y].getDeviation();
                }
            }
            if (estimator != null) {
                for (int i = 0; i < forecastHorizon; i++) {
                    int[] missing = new int[inputLength];
                    query[inputLength] = i;
                    for (int j = 0; j < inputLength; j++) {
                        missing[j] = inputLength + 1 + j;
                    }
                    // at this moment we use the PredictiveRCF more for the shorter term estimation,
                    // and use an interpolation
                    // with the observed error for the longer term
                    SampleSummary answer = estimator.predict(query, 0, missing, 1, 0.5, 1.0, false);
                    for (int j = 0; j < inputLength; j++) {
                        errorHigh[i * inputLength + j] = (forecastHorizon - j)
                                * max(0, answer.median[inputLength + 1 + j]) / forecastHorizon
                                + (float) (j * rmseHighDeviations[i * inputLength + j].getDeviation()
                                        / forecastHorizon);
                    }
                    for (int j = 0; j < inputLength; j++) {
                        missing[j] = 2 * inputLength + 1 + j;
                    }
                    answer = estimator.predict(query, 0, missing, 1, 0.5, 1.0, false);
                    for (int j = 0; j < inputLength; j++) {
                        errorLow[i * inputLength + j] = (forecastHorizon - j)
                                * max(0, -answer.median[2 * inputLength + 1 + j]) / forecastHorizon
                                + (float) (j * rmseLowDeviations[i * inputLength + j].getDeviation() / forecastHorizon);
                    }
                }
                for (int y = 0; y < intervalPrecision.length; y++) {
                    errorHigh[y] = (float) max(1.0, 1.0 / (intervalPrecision[y].getMean() + 0.1)) * errorHigh[y];
                    errorLow[y] = (float) max(1.0, 1.0 / (intervalPrecision[y].getMean() + 0.1)) * errorLow[y];
                }
            } else {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    errorHigh[y] = (float) errorRMSE.high[y];
                    errorLow[y] = (float) errorRMSE.low[y];
                }
            }

        }
        for (int i = 0; i < errorMean.length; i++) {
            errorMean[i] = (float) (rmseHighDeviations[i].getMean() + rmseLowDeviations[i].getMean());
            errorDistribution.values[i] = errorMean[i];
            errorDistribution.upper[i] = errorMean[i] + (float) (1.3 * errorHigh[i]);
            errorDistribution.lower[i] = errorMean[i] - (float) (1.3 * errorLow[i]);
        }
    }

    public void augmentDescriptor(ForecastDescriptor descriptor) {
        int inputLength = descriptor.getInputLength();
        float[] iPrecision = new float[inputLength * forecastHorizon];
        for (int i = 0; i < errorMean.length; i++) {
            iPrecision[i] = (float) intervalPrecision[i].getMean();
        }
        descriptor.setErrorMean(errorMean);
        descriptor.setErrorRMSE(errorRMSE);
        descriptor.setObservedErrorDistribution(errorDistribution);
        descriptor.setIntervalPrecision(iPrecision);
    }

    /**
     * saves the forecast -- note that this section assumes that updateActuals() has
     * been invoked prior (to recompute the deviations)
     * 
     * @param vector the forecast
     */
    public void updateForecasts(RangeVector vector) {
        int arrayLength = pastForecasts.length;
        int storedForecastIndex = (sequenceIndex + arrayLength - 1) % (arrayLength);
        int length = pastForecasts[0].values.length;
        System.arraycopy(vector.values, 0, pastForecasts[storedForecastIndex].values, 0, length);
        System.arraycopy(vector.upper, 0, pastForecasts[storedForecastIndex].upper, 0, length);
        System.arraycopy(vector.lower, 0, pastForecasts[storedForecastIndex].lower, 0, length);
    }

    public RangeVector getErrorDistribution() {
        return new RangeVector(errorDistribution);
    }

    public float[] getErrorMean() {
        return Arrays.copyOf(errorMean, errorMean.length);
    }

    public DiVector getErrorRMSE() {
        return new DiVector(errorRMSE);
    }

    public Deviation[] getDeviationList() {
        Deviation[] list = new Deviation[3 * intervalPrecision.length];
        for (int i = 0; i < intervalPrecision.length; i++) {
            list[i] = intervalPrecision[i].copy();
            list[i + intervalPrecision.length] = rmseHighDeviations[i].copy();
            list[i + 2 * intervalPrecision.length] = rmseLowDeviations[i].copy();
        }
        return list;
    }

    public float[] getIntervalPrecision() {
        float[] iPrecision = new float[intervalPrecision.length];
        for (int i = 0; i < iPrecision.length; i++) {
            iPrecision[i] = (float) (intervalPrecision[i].getMean());
        }
        return iPrecision;
    }

    public void calibrate(double[] input, Calibration calibration, RangeVector ranges) {
        if (calibration != Calibration.NONE) {
            int inputLength = intervalPrecision.length / forecastHorizon;
            checkArgument(input.length == inputLength, "incorrect input");
            checkArgument(intervalPrecision.length == ranges.values.length, "mismatched lengths");
            for (int y = 0; y < intervalPrecision.length; y++) {
                if (calibration == Calibration.SIMPLE) {
                    ranges.values[y] = min(
                            max(ranges.values[y] + errorDistribution.values[y], lowerLimit[y % inputLength]),
                            upperLimit[y % inputLength]);
                } else {
                    ranges.values[y] = min(max(ranges.values[y], lowerLimit[y % inputLength]),
                            upperLimit[y % inputLength]);
                }
                ranges.upper[y] = min(max(ranges.upper[y], ranges.values[y] + errorDistribution.upper[y]),
                        upperLimit[y % inputLength]);
                ranges.lower[y] = max(min(ranges.lower[y], ranges.values[y] + errorDistribution.lower[y]),
                        lowerLimit[y % inputLength]);
            }
        }
    }

    public float[] getPastForecastsFlattened() {
        int arrayLength = min(sequenceIndex, pastForecasts.length);
        int length = intervalPrecision.length;
        float[] answer = new float[3 * length * arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            System.arraycopy(pastForecasts[i].values, 0, answer, 3 * i * length, length);
            System.arraycopy(pastForecasts[i].upper, 0, answer, 3 * i * length + length, length);
            System.arraycopy(pastForecasts[i].lower, 0, answer, 3 * i * length + 2 * length, length);
        }
        return answer;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        protected int dimensions;
        protected int shingleSize = 1;
        protected int forecastHorizon;
        protected boolean useRCF = true;
        protected int errorHorizon = 100; // easy for percentile
        protected double percentile = DEFAULT_ERROR_PERCENTILE;
        protected Optional<float[]> upperLimit = Optional.empty();
        protected Optional<float[]> lowerLimit = Optional.empty();

        public Builder dimensions(int dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        public Builder shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return this;
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

        public Builder lowerLimit(float[] lowerLimit) {
            this.lowerLimit = Optional.of(lowerLimit);
            return this;
        }

        public Builder upperLimit(float[] upperLimit) {
            this.upperLimit = Optional.of(upperLimit);
            return this;
        }

        public Builder useRCF(boolean use) {
            useRCF = use;
            return this;
        }

        public ErrorHandler build() {
            return new ErrorHandler(this);
        }
    }
}
