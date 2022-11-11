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
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

public class ErrorHandler {

    /**
     * this to constrain the state
     */
    public static int MAX_ERROR_HORIZON = 1024;

    int sequenceIndex;
    double percentile;
    int forecastHorizon;
    int errorHorizon;
    protected RangeVector[] pastForecasts;
    protected float[][] actuals;
    // the following are derived quantities and present for efficiency reasons
    RangeVector errorDistribution;
    DiVector errorRMSE;
    float[] errorMean;
    DiVector multipliers;
    float[] intervalPrecision;

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
        multipliers = new DiVector(length);
        Arrays.fill(multipliers.high, 1);
        Arrays.fill(multipliers.low, 1);
        intervalPrecision = new float[length];
        errorDistribution = new RangeVector(length);
        Arrays.fill(errorDistribution.upper, Float.MAX_VALUE);
        Arrays.fill(errorDistribution.lower, Float.MIN_VALUE);
    }

    /**
     * the folloqing would be useful when states and mappers get written
     */
    public ErrorHandler(int errorHorizon, int forecastHorizon, int sequenceIndex, double percentile, int inputLength,
            int dimensions, float[] actualsFlattened, float[] pastForecastsFlattened, float[] auxilliary) {
        checkArgument(forecastHorizon > 0, " incorrect forecast horizon");
        checkArgument(errorHorizon >= forecastHorizon, "incorrect error horizon");
        checkArgument(actualsFlattened != null || pastForecastsFlattened == null,
                " actuals and forecasts are a mismatch");
        checkArgument(inputLength > 0 && dimensions > 0 && dimensions % inputLength == 0, "incorrect parameters");
        this.sequenceIndex = sequenceIndex;
        this.errorHorizon = errorHorizon;
        this.percentile = percentile;
        this.forecastHorizon = forecastHorizon;
        int currentLength = (actualsFlattened == null) ? 0 : actualsFlattened.length;
        checkArgument(currentLength % inputLength == 0, "actuals array is incorrect");
        int forecastLength = (pastForecastsFlattened == null) ? 0 : pastForecastsFlattened.length;
        checkArgument(forecastLength == currentLength * dimensions * 3 / inputLength, "misaligned forecasts");
        int arrayLength = max(forecastHorizon + errorHorizon, currentLength / inputLength);
        this.pastForecasts = new RangeVector[arrayLength];
        this.actuals = new float[arrayLength][inputLength];

        int length = forecastHorizon * inputLength;

        this.errorMean = new float[length];
        this.errorRMSE = new DiVector(length);
        this.intervalPrecision = new float[length];
        this.multipliers = new DiVector(length);
        this.errorDistribution = new RangeVector(length);

        if (pastForecastsFlattened != null) {
            for (int i = 0; i < currentLength / inputLength; i++) {
                float[] values = Arrays.copyOfRange(pastForecastsFlattened, i * 3 * dimensions,
                        (i * 3 + 1) * dimensions);
                float[] upper = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 1) * dimensions,
                        (i * 3 + 2) * dimensions);
                float[] lower = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 3) * dimensions,
                        (i * 3 + 3) * dimensions);
                pastForecasts[i] = new RangeVector(values, upper, lower);
                System.arraycopy(actualsFlattened, i * inputLength, actuals[i], 0, inputLength);
            }
            calibrate();
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
        int errorIndex = sequenceIndex % (arrayLength);
        int inputLength = descriptor.getInputLength();
        double[] input = descriptor.getCurrentInput();

        for (int i = 0; i < inputLength; i++) {
            actuals[errorIndex][i] = (float) input[i];
        }
        ++sequenceIndex;
        calibrate();
        if (calibrationMethod != Calibration.NONE) {
            // adjusting intervals based on past performance always seem to be a good idea
            for (int i = 0; i < inputLength * forecastHorizon; i++) {
                descriptor.timedForecast.rangeVector.upper[i] = max(descriptor.timedForecast.rangeVector.upper[i],
                        descriptor.timedForecast.rangeVector.values[i] + errorDistribution.upper[i]);
                descriptor.timedForecast.rangeVector.lower[i] = min(descriptor.timedForecast.rangeVector.lower[i],
                        descriptor.timedForecast.rangeVector.values[i] + errorDistribution.lower[i]);
            }
            // changing the forecast has actual consequences to the error; and since this is
            // feedback system
            // should be done with deliberation
            if (calibrationMethod == Calibration.SIMPLE) {
                for (int i = 0; i < inputLength * forecastHorizon; i++) {
                    descriptor.timedForecast.rangeVector.values[i] += errorDistribution.values[i];
                }
            }
        }
        descriptor.setErrorMean(errorMean);
        descriptor.setErrorRMSE(errorRMSE);
        descriptor.setObservedErrorDistribution(errorDistribution);
        descriptor.setCalibration(intervalPrecision);
        System.arraycopy(descriptor.timedForecast.rangeVector.values, 0, pastForecasts[errorIndex].values, 0, length);
        System.arraycopy(descriptor.timedForecast.rangeVector.upper, 0, pastForecasts[errorIndex].upper, 0, length);
        System.arraycopy(descriptor.timedForecast.rangeVector.lower, 0, pastForecasts[errorIndex].lower, 0, length);

    }

    public RangeVector getErrors() {
        return new RangeVector(errorDistribution);
    }

    public float[] getErrorMean() {
        return Arrays.copyOf(errorMean, errorMean.length);
    }

    public DiVector getErrorRMSE() {
        return new DiVector(errorRMSE);
    }

    public float[] getCalibration() {
        return Arrays.copyOf(intervalPrecision, intervalPrecision.length);
    }

    public DiVector getMultipliers() {
        return new DiVector(multipliers);
    }

    public RangeVector computeErrorPercentile(double percentile, BiFunction<Float, Float, Float> error) {
        return computeErrorPercentile(percentile, pastForecasts.length, error);
    }

    /**
     * the following function is provided such that the calibration of errors can be
     * performed using a different function. e.g., SMAPE type evaluation using
     * RCFCaster.alternateError
     * 
     * @param percentile the desired percentile (we recomment leaving this at 0.1 --
     *                   the algorithm likely will never have sufficiently many
     *                   observations to have a very fine grain distribution;
     *                   moreover any forecasting should concentrate the measure (no
     *                   pun intended) towards 0.5 and for a successful forecaster
     *                   these error percentiles will be less meaningful
     * @param newHorizon a horizon which could be smaller than the errorHorizon (for
     *                   auditing)
     * @param error      the function that defies error
     * @return a RangeVector where the value field corresponds to the p50 and the
     *         upper and lower arrays correspond to percentile and 1-percentile
     *         quantiles. We reiterate that do be cautious in use of percentiles.
     *         These are observational quantiles and the act of interpreting them as
     *         probabilities is riddled with assumptions
     */
    public RangeVector computeErrorPercentile(double percentile, int newHorizon,
            BiFunction<Float, Float, Float> error) {
        checkArgument(newHorizon <= errorHorizon && newHorizon > 0, "incorrect horizon parameter");
        int length = pastForecasts[0].values.length;
        float[] lower = new float[length];
        float[] upper = new float[length];
        float[] values = new float[length];
        Arrays.fill(lower, -Float.MAX_VALUE);
        Arrays.fill(upper, Float.MAX_VALUE);
        if (actuals != null) {
            int inputLength = actuals[0].length;
            for (int i = 0; i < forecastHorizon; i++) {
                // this is the only place where the newer (possibly shorter) horizon matters
                int len = (sequenceIndex > newHorizon + i + 1) ? newHorizon : sequenceIndex - i - 1;

                for (int j = 0; j < inputLength; j++) {
                    int pos = i * inputLength + j;
                    if (len > 0) {
                        double[] copy = getErrorVector(len, (i + 1), j, pos, error);
                        double fracRank = percentile * len;
                        Arrays.sort(copy);
                        values[pos] = interpolatedMedian(copy);
                        lower[pos] = interpolatedLowerRank(copy, fracRank);
                        upper[pos] = interpolatedUpperRank(copy, len, fracRank);
                    }
                }
            }
        }
        return new RangeVector(values, upper, lower);
    }

    protected double[] getErrorVector(int len, int leadtime, int inputCoordinate, int position,
            BiFunction<Float, Float, Float> error) {
        int arrayLength = pastForecasts.length;
        int errorIndex = (sequenceIndex - 1 + arrayLength) % arrayLength;
        double[] copy = new double[len];
        for (int k = 0; k < len; k++) {
            int pastIndex = (errorIndex - leadtime - k + arrayLength) % arrayLength;
            int index = (errorIndex - k + arrayLength) % arrayLength;
            copy[k] = error.apply(actuals[index][inputCoordinate], pastForecasts[pastIndex].values[position]);
        }
        return copy;
    }

    /*
     * this method computes a lot of different quantities, some of which would be
     * useful in the future.
     */
    protected void calibrate() {
        int inputLength = actuals[0].length;
        int arrayLength = pastForecasts.length;
        int errorIndex = (sequenceIndex - 1 + arrayLength) % arrayLength;
        double[] copy = new double[errorHorizon];
        Double[] errorintervals = new Double[errorHorizon];
        Arrays.fill(intervalPrecision, 0);
        for (int i = 0; i < forecastHorizon; i++) {
            // this is the only place where the newer (possibly shorter) horizon matters
            int len = (sequenceIndex > errorHorizon + i + 1) ? errorHorizon : sequenceIndex - i - 1;

            for (int j = 0; j < inputLength; j++) {
                int pos = i * inputLength + j;
                if (len > 0) {
                    double positiveSum = 0;
                    int positiveCount = 0;
                    double negativeSum = 0;
                    double positiveSqSum = 0;
                    double negativeSqSum = 0;
                    for (int k = 0; k < len; k++) {
                        int pastIndex = (errorIndex - (i + 1) - k + arrayLength) % arrayLength;
                        int index = (errorIndex - k + arrayLength) % arrayLength;
                        intervalPrecision[pos] += (actuals[index][j] <= pastForecasts[pastIndex].upper[pos]
                                && actuals[index][j] >= pastForecasts[pastIndex].lower[pos]) ? 1 : 0;
                        double error = actuals[index][j] - pastForecasts[pastIndex].values[pos];
                        if (error >= 0) {
                            positiveSum += error;
                            positiveSqSum += error * error;
                            ++positiveCount;
                            double t = actuals[index][j] - pastForecasts[pastIndex].upper[pos];
                            errorintervals[k] = (t <= 0 || t >= error) ? 1.0 : error / (error - t);
                        } else {
                            negativeSum += error;
                            negativeSqSum += error * error;
                            double t = actuals[index][j] - pastForecasts[pastIndex].lower[pos];
                            errorintervals[k] = (t >= 0 || t <= error) ? 1.0 : error / (t - error);
                        }
                        copy[k] = error;
                    }
                    errorMean[pos] = (float) (positiveSum + negativeSum) / len;
                    errorRMSE.high[pos] = (positiveCount > 0) ? Math.sqrt(positiveSqSum / positiveCount) : 0;
                    errorRMSE.low[pos] = (positiveCount < len) ? -Math.sqrt(negativeSqSum / (len - positiveCount)) : 0;
                    Arrays.sort(copy, 0, len);
                    errorDistribution.values[pos] = interpolatedMedian(copy);
                    errorDistribution.upper[pos] = interpolatedUpperRank(copy, len, len * percentile);
                    errorDistribution.lower[pos] = interpolatedLowerRank(copy, len * percentile);
                    Arrays.sort(errorintervals, 0, len, (x, y) -> Double.compare(abs((double) y), abs((double) (x))));
                    double t = errorintervals[(int) Math.floor(RCFCaster.DEFAULT_ERROR_PERCENTILE * len)];
                    double positive = 0;
                    int posCount = 0;
                    double negative = 0;
                    int negCount = 0;
                    for (int k = 0; k < len; k++) {
                        if (errorintervals[k] < abs(t) && errorintervals[k] > 0) {
                            positive += errorintervals[k];
                            posCount++;
                        }
                        if (-errorintervals[k] < abs(t) && errorintervals[k] < 0) {
                            negative += -errorintervals[k];
                            negCount++;
                        }
                    }
                    multipliers.high[pos] = (posCount > 0) ? positive / posCount : 1;
                    multipliers.low[pos] = (negCount > 0) ? negative / negCount : 1;
                    intervalPrecision[pos] = intervalPrecision[pos] / len;
                } else {
                    errorMean[pos] = 0;
                    errorRMSE.high[pos] = errorRMSE.low[pos] = 0;
                    errorDistribution.values[pos] = 0;
                    errorDistribution.upper[pos] = Float.MAX_VALUE;
                    errorDistribution.lower[pos] = -Float.MAX_VALUE;
                    multipliers.high[pos] = multipliers.low[pos] = 1;
                    intervalPrecision[pos] = 0;
                }
            }
        }
    }

    float interpolatedMedian(double[] array) {
        checkArgument(array != null, " cannot be null");
        int len = array.length;
        if (len % 2 != 0) {
            return (float) array[len / 2];
        } else {
            return (float) ((array[len / 2 - 1] + array[len / 2]) / 2);
        }
    }

    float interpolatedLowerRank(double[] array, double fracRank) {
        if (fracRank < 1) {
            return -Float.MAX_VALUE;
        }
        int rank = (int) Math.floor(fracRank);
        if (!RCFCaster.USE_INTERPOLATION_IN_DISTRIBUTION) {
            // turn off interpolation
            fracRank = rank;
        }
        return (float) (array[rank - 1] + (fracRank - rank) * (array[rank] - array[rank - 1]));
    }

    float interpolatedUpperRank(double[] array, int len, double fracRank) {
        if (fracRank < 1) {
            return Float.MAX_VALUE;
        }
        int rank = (int) Math.floor(fracRank);
        if (!RCFCaster.USE_INTERPOLATION_IN_DISTRIBUTION) {
            // turn off interpolation
            fracRank = rank;
        }
        return (float) (array[len - rank] + (fracRank - rank) * (array[len - rank - 1] - array[len - rank]));
    }

}
