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
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.function.BiFunction;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

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

    /**
     * this to constrain the state
     */
    public static int MAX_ERROR_HORIZON = 1024;

    int sequenceIndex;
    double percentile;
    int forecastHorizon;
    int errorHorizon;
    // the following arrays store the state of the sequential computation
    // these can be optimized -- for example once could store the errors; which
    // would see much fewer
    // increments. However for a small enough errorHorizon, the genrality of
    // changing the error function
    // outweighs the benefit of recomputation. The search in the ensemble tree is
    // still a larger bottleneck than
    // these computations at the moment; not to mention issues of saving and
    // restoring state.
    protected RangeVector[] pastForecasts;
    protected float[][] actuals;
    // the following are derived quantities and present for efficiency reasons
    RangeVector errorDistribution;
    DiVector errorRMSE;
    float[] errorMean;
    float[] intervalPrecision;

    // We keep the multiplers defined for potential
    // future use.

    RangeVector multipliers;
    RangeVector adders;

    public ErrorHandler(RCFCaster.Builder builder) {
        checkArgument(builder.errorHorizon >= builder.forecastHorizon,
                "intervalPrecision horizon should be at least as large as forecast horizon");
        checkArgument(builder.errorHorizon <= MAX_ERROR_HORIZON, "reduce error horizon of change MAX");
        forecastHorizon = builder.forecastHorizon;
        errorHorizon = builder.errorHorizon;
        int inputLength = (builder.dimensions / builder.shingleSize);
        int length = inputLength * forecastHorizon;
        percentile = builder.percentile;
        pastForecasts = new RangeVector[errorHorizon + forecastHorizon];
        for (int i = 0; i < errorHorizon + forecastHorizon; i++) {
            pastForecasts[i] = new RangeVector(length);
        }
        actuals = new float[errorHorizon + forecastHorizon][inputLength];
        sequenceIndex = 0;
        errorMean = new float[length];
        errorRMSE = new DiVector(length);
        multipliers = new RangeVector(length);
        Arrays.fill(multipliers.upper, 1);
        Arrays.fill(multipliers.lower, 1);
        adders = new RangeVector(length);
        intervalPrecision = new float[length];
        errorDistribution = new RangeVector(length);
        Arrays.fill(errorDistribution.upper, Float.MAX_VALUE);
        Arrays.fill(errorDistribution.lower, -Float.MAX_VALUE);
    }

    // for mappers
    public ErrorHandler(int errorHorizon, int forecastHorizon, int sequenceIndex, double percentile, int inputLength,
            float[] actualsFlattened, float[] pastForecastsFlattened, float[] auxilliary) {
        checkArgument(forecastHorizon > 0, " incorrect forecast horizon");
        checkArgument(errorHorizon >= forecastHorizon, "incorrect error horizon");
        checkArgument(actualsFlattened != null || pastForecastsFlattened == null,
                " actuals and forecasts are a mismatch");
        checkArgument(inputLength > 0, "incorrect parameters");
        // calibration would have been performed at previous value
        this.sequenceIndex = sequenceIndex - 1;
        this.errorHorizon = errorHorizon;
        this.percentile = percentile;
        this.forecastHorizon = forecastHorizon;
        int currentLength = (actualsFlattened == null) ? 0 : actualsFlattened.length;
        checkArgument(currentLength % inputLength == 0, "actuals array is incorrect");
        int forecastLength = (pastForecastsFlattened == null) ? 0 : pastForecastsFlattened.length;

        int arrayLength = max(forecastHorizon + errorHorizon, currentLength / inputLength);
        this.pastForecasts = new RangeVector[arrayLength];
        this.actuals = new float[arrayLength][inputLength];

        int length = forecastHorizon * inputLength;
        // currentLength = (number of actual time steps stored) x inputLength and for
        // each of the stored time steps we get a forecast whose length is
        // forecastHorizon x inputLength (and then upper and lower for each, hence x 3)
        // so forecastLength = number of actual time steps stored x forecastHorizon x
        // inputLength x 3
        // = currentLength x forecastHorizon x 3
        checkArgument(forecastLength == currentLength * 3 * forecastHorizon, "misaligned forecasts");

        this.errorMean = new float[length];
        this.errorRMSE = new DiVector(length);
        this.intervalPrecision = new float[length];
        this.adders = new RangeVector(length);
        this.multipliers = new RangeVector(length);
        this.errorDistribution = new RangeVector(length);

        if (pastForecastsFlattened != null) {
            for (int i = 0; i < arrayLength; i++) {
                float[] values = Arrays.copyOfRange(pastForecastsFlattened, i * 3 * length, (i * 3 + 1) * length);
                float[] upper = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 1) * length, (i * 3 + 2) * length);
                float[] lower = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 2) * length, (i * 3 + 3) * length);
                pastForecasts[i] = new RangeVector(values, upper, lower);
                System.arraycopy(actualsFlattened, i * inputLength, actuals[i], 0, inputLength);
            }
            recomputeErrorsAndCalibrate(Calibration.NONE, null, null);
            ++this.sequenceIndex;
        }
    }

    /**
     * the following the core subroutine, which calibrates; but the application of
     * the calibration is controlled
     *
     * @param descriptor        the current forecast
     * @param calibrationMethod the choice of the callibration
     */

    public void update(ForecastDescriptor descriptor, Calibration calibrationMethod) {
        int arrayLength = pastForecasts.length;
        int length = pastForecasts[0].values.length;
        int storedForecastIndex = sequenceIndex % (arrayLength);
        int inputLength = descriptor.getInputLength();
        double[] input = descriptor.getCurrentInput();

        if (sequenceIndex > 0) {
            // sequenceIndex indicates the first empty place for input
            // note the predictions have already been stored
            int inputIndex = (sequenceIndex + arrayLength - 1) % arrayLength;
            for (int i = 0; i < inputLength; i++) {
                actuals[inputIndex][i] = (float) input[i];
            }
        }

        recomputeErrorsAndCalibrate(calibrationMethod, descriptor.getPostDeviations(),
                descriptor.timedForecast.rangeVector);

        descriptor.setErrorMean(errorMean);
        descriptor.setErrorRMSE(errorRMSE);
        descriptor.setObservedErrorDistribution(errorDistribution);
        descriptor.setIntervalPrecision(intervalPrecision);

        System.arraycopy(descriptor.timedForecast.rangeVector.values, 0, pastForecasts[storedForecastIndex].values, 0,
                length);
        System.arraycopy(descriptor.timedForecast.rangeVector.upper, 0, pastForecasts[storedForecastIndex].upper, 0,
                length);
        System.arraycopy(descriptor.timedForecast.rangeVector.lower, 0, pastForecasts[storedForecastIndex].lower, 0,
                length);
        ++sequenceIndex;
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

    public float[] getIntervalPrecision() {
        return Arrays.copyOf(intervalPrecision, intervalPrecision.length);
    }

    public RangeVector getMultipliers() {
        return new RangeVector(multipliers);
    }

    public RangeVector getAdders() {
        return new RangeVector(adders);
    }

    protected double[] getErrorVector(int len, int leadtime, int inputCoordinate, int position,
            BiFunction<Float, Float, Float> error) {
        int arrayLength = pastForecasts.length;
        int errorIndex = (sequenceIndex - 1 + arrayLength) % arrayLength;
        double[] copy = new double[len];
        for (int k = 0; k < len; k++) {
            int pastIndex = (errorIndex - leadtime - k + arrayLength) % arrayLength;
            int index = (errorIndex - k - 1 + arrayLength) % arrayLength;
            copy[k] = error.apply(actuals[index][inputCoordinate], pastForecasts[pastIndex].values[position]);
        }
        return copy;
    }

    /*
     * this method computes the errors and performs calibration. It is done together
     * to avoid recreating the error arrays In particular it splits the RMSE into
     * positive and negative contribution which is informative about directionality
     * of error.
     */
    protected void recomputeErrorsAndCalibrate(Calibration calibration, double[] errorDeviations, RangeVector ranges) {
        int inputLength = actuals[0].length;
        int arrayLength = pastForecasts.length;
        int inputIndex = (sequenceIndex - 1 + arrayLength) % arrayLength;
        double[] medianError = new double[errorHorizon];

        Arrays.fill(intervalPrecision, 0);
        for (int i = 0; i < forecastHorizon; i++) {
            // this is the only place where the newer (possibly shorter) horizon matters
            int len = (sequenceIndex > errorHorizon + i) ? errorHorizon : sequenceIndex - i;

            for (int j = 0; j < inputLength; j++) {
                int pos = i * inputLength + j;
                if (len > 0) {
                    double positiveSum = 0;
                    int positiveCount = 0;
                    double negativeSum = 0;
                    double positiveSqSum = 0;
                    double negativeSqSum = 0;
                    for (int k = 0; k < len; k++) {
                        int pastIndex = (inputIndex - i - k + arrayLength) % arrayLength;
                        // one more forecast
                        int index = (inputIndex - k + arrayLength) % arrayLength;
                        double error = actuals[index][j] - pastForecasts[pastIndex].values[pos];
                        medianError[k] = error;
                        intervalPrecision[pos] += (pastForecasts[pastIndex].upper[pos] >= actuals[index][j]
                                && actuals[index][j] >= pastForecasts[pastIndex].lower[pos]) ? 1 : 0;

                        if (error >= 0) {
                            positiveSum += error;
                            positiveSqSum += error * error;
                            ++positiveCount;
                        } else {
                            negativeSum += error;
                            negativeSqSum += error * error;
                        }
                    }
                    errorMean[pos] = (float) (positiveSum + negativeSum) / len;
                    errorRMSE.high[pos] = (positiveCount > 0) ? Math.sqrt(positiveSqSum / positiveCount) : 0;
                    errorRMSE.low[pos] = (positiveCount < len) ? -Math.sqrt(negativeSqSum / (len - positiveCount)) : 0;

                    if (len * percentile >= 1.0) {
                        Arrays.sort(medianError, 0, len);
                        // medianError array is now sorted
                        errorDistribution.values[pos] = interpolatedMedian(medianError, len);
                        errorDistribution.upper[pos] = interpolatedUpperRank(medianError, len, len * percentile);
                        errorDistribution.lower[pos] = interpolatedLowerRank(medianError, len * percentile);
                    }
                    intervalPrecision[pos] = intervalPrecision[pos] / len;
                } else {
                    errorMean[pos] = 0;
                    errorRMSE.high[pos] = errorRMSE.low[pos] = 0;
                    errorDistribution.values[pos] = errorDistribution.upper[pos] = errorDistribution.lower[pos] = 0;
                    adders.upper[pos] = adders.lower[pos] = adders.values[pos] = 0;
                    intervalPrecision[pos] = 0;
                }
                if (ranges != null && calibration != Calibration.NONE) {
                    if (len * percentile < 1.0) {
                        double deviation = (errorDeviations == null) ? 0 : errorDeviations[j];
                        ranges.upper[pos] = max(ranges.upper[pos], ranges.values[pos] + (float) (1.3 * deviation));
                        ranges.lower[pos] = min(ranges.lower[pos], ranges.values[pos] - (float) (1.3 * deviation));
                    } else {
                        if (calibration == Calibration.SIMPLE) {
                            adjust(ranges, errorDistribution);
                        }
                        if (calibration == Calibration.MINIMAL) {
                            adjustMinimal(ranges, errorDistribution);
                        }
                    }
                }
            }
        }
    }

    protected float interpolatedMedian(double[] ascendingArray, int len) {
        checkArgument(ascendingArray != null, " cannot be null");
        checkArgument(ascendingArray.length >= len, "incorrect length parameter");
        float lower = (float) ((len % 2 == 0) ? ascendingArray[len / 2 - 1]
                : (ascendingArray[len / 2] + ascendingArray[len / 2 - 1]) / 2);
        float upper = (float) ((len % 2 == 0) ? ascendingArray[len / 2]
                : (ascendingArray[len / 2] + ascendingArray[len / 2 + 1]) / 2);
        if (lower <= 0 && 0 <= upper) {
            // 0 is plausible, and introduces minimal externality
            return 0;
        } else {
            return (upper + lower) / 2;
        }
    }

    float interpolatedLowerRank(double[] ascendingArray, double fracRank) {
        int rank = (int) Math.floor(fracRank);
        return (float) (ascendingArray[rank - 1]
                + (fracRank - rank) * (ascendingArray[rank] - ascendingArray[rank - 1]));
    }

    float interpolatedUpperRank(double[] ascendingArray, int len, double fracRank) {
        int rank = (int) Math.floor(fracRank);
        return (float) (ascendingArray[len - rank]
                + (fracRank - rank) * (ascendingArray[len - rank - 1] - ascendingArray[len - rank]));
    }

    void adjust(RangeVector rangeVector, RangeVector other) {
        checkArgument(other.values.length == rangeVector.values.length, " mismatch in lengths");
        for (int i = 0; i < rangeVector.values.length; i++) {
            rangeVector.values[i] += other.values[i];
            rangeVector.upper[i] = max(rangeVector.values[i], rangeVector.upper[i] + other.upper[i]);
            rangeVector.lower[i] = min(rangeVector.values[i], rangeVector.lower[i] + other.lower[i]);
        }
    }

    void adjustMinimal(RangeVector rangeVector, RangeVector other) {
        checkArgument(other.values.length == rangeVector.values.length, " mismatch in lengths");
        for (int i = 0; i < rangeVector.values.length; i++) {
            float oldVal = rangeVector.values[i];
            rangeVector.values[i] += other.values[i];
            rangeVector.upper[i] = max(rangeVector.values[i], oldVal + other.upper[i]);
            rangeVector.lower[i] = min(rangeVector.values[i], oldVal + other.lower[i]);
        }
    }
}
