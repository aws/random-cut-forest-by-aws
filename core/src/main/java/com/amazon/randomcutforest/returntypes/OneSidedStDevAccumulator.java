/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.returntypes;

/**
 * This accumulator checks to see if a result is converging by testing the
 * sample mean and standard deviation of a scalar value computed from the
 * result. As the name implies, the accumulator performs a one-sided check,
 * comparing the new value the current sample mean and updating its converged
 * status only if the difference is positive (or negative, if highIsCritical is
 * set to false. This accumulator is intended to be used with values where we
 * care more about outliers in one direction. For example, if our statistic is
 * anomaly score, we are normally more concerned with high anomaly scores than
 * low ones.
 *
 * @param <R> The type of the value being accumulated.
 */
public abstract class OneSidedStDevAccumulator<R> implements ConvergingAccumulator<R> {

    /**
     * When testing for convergence, we use ALPHA times the sample standard
     * deviation to define our interval.
     */
    private static final double ALPHA = 0.5;
    /**
     * The minimum number of values that have to be accepted by this accumulator
     * before we start testing for convergence.
     */
    private final int minValuesAccepted;
    /**
     * The number of witnesses needed to declare convergence.
     */
    private final int convergenceThreshold;
    /**
     * Set to 'true' if we care more about high values of the converging scalar than
     * low values. Set to 'false' if the opposite is true.
     */
    private final boolean highIsCritical;
    /**
     * This value is +1 if highIsCritical is 'true', and -1 if highIsCritical is
     * fault. It is used in the converegence test.
     */
    private final int sign;
    /**
     * The value accumulated until now.
     */
    protected R accumulatedValue;
    /**
     * The number of values accepted by this accumulator until now.
     */
    private int valuesAccepted;
    /**
     * The number of values that are 'witnesses' to convergence until now. See
     * {@link #accept}.
     */
    private int witnesses;
    /**
     * The current sum of the converging scalar value. Used to compute the sample
     * mean.
     */
    private double sumConvergeVal;
    /**
     * The current sum of squares of the converging scalar value. Used to compute
     * the sample standard deviation.
     */
    private double sumSqConvergeVal;

    /**
     * Create a new converging accumulator that uses a one-sided standard deviation
     * test.
     *
     * @param highIsCritical    Set to 'true' if we care more about high values of
     *                          the converging scalar than low values. Set to
     *                          'false' if the opposite is true.
     * @param precision         The number of witnesses required before declaring
     *                          convergence will be at least 1.0 / precision.
     * @param minValuesAccepted The user-specified minimum number of values visited
     *                          before returning a result. Note that
     *                          {@link #isConverged()} may return true before
     *                          accepting this number of results if the
     * @param maxValuesAccepted The maximum number of values that will be accepted
     *                          by this accumulator.
     */
    public OneSidedStDevAccumulator(boolean highIsCritical, double precision, int minValuesAccepted,
            int maxValuesAccepted) {

        this.highIsCritical = highIsCritical;
        this.convergenceThreshold = precision < 1.0 / maxValuesAccepted ? maxValuesAccepted : (int) (1.0 / precision);
        this.minValuesAccepted = Math.min(minValuesAccepted, maxValuesAccepted);
        valuesAccepted = 0;
        witnesses = 0;
        sumConvergeVal = 0.0;
        sumSqConvergeVal = 0.0;
        sign = highIsCritical ? 1 : -1;
        accumulatedValue = null;
    }

    /**
     * Given a new result value, add it to the accumulated value and update
     * convergence statistics.
     *
     * @param result The new value being accumulated.
     */
    @Override
    public void accept(R result) {
        accumulateValue(result);
        double value = getConvergingValue(result);
        sumConvergeVal += value;
        sumSqConvergeVal += value * value;
        valuesAccepted++;

        if (valuesAccepted >= minValuesAccepted) {
            // note that using the last seen value in the deviation dampens its effect

            double mean = sumConvergeVal / valuesAccepted;
            double stdev = sumSqConvergeVal / valuesAccepted - mean * mean;

            stdev = stdev < 0 ? 0 : Math.sqrt(stdev);

            if (sign * (value - mean) > ALPHA * stdev) {
                witnesses++;
            }
        }
    }

    /**
     * @return the number of values accepted until now.
     */
    @Override
    public int getValuesAccepted() {
        return valuesAccepted;
    }

    /**
     * @return 'true' if the accumulated value has converged, 'false' otherwise.
     */
    @Override
    public boolean isConverged() {
        return witnesses >= convergenceThreshold;
    }

    /**
     * @return the accumulated value.
     */
    @Override
    public R getAccumulatedValue() {
        return accumulatedValue;
    }

    /**
     * Given a new result value, compute its converging scalar value.
     *
     * @param result A new result value computed by a Random Cut Tree.
     * @return the scalar value used to measure convergence for this result type.
     */
    protected abstract double getConvergingValue(R result);

    /**
     * Add the new result to the accumulated value.
     *
     * @param result The new result to add to the accumulated value.
     */
    protected abstract void accumulateValue(R result);
}
