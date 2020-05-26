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

package com.amazon.randomcutforest.returntypes;

/**
 * A converging accumulator using a one-sided standard deviation tests. The
 * accumulator tests the sum of entries (i.e., the "high-low sum") in the
 * submitted DiVectors for convergence and returns the sum of all submitted
 * DiVectors.
 */
public class OneSidedConvergingDiVectorAccumulator extends OneSidedStDevAccumulator<DiVector> {

    /**
     * Create a new converging accumulator that uses a one-sided standard deviation
     * test.
     *
     * @param dimensions        The number of dimensions in the DiVectors being
     *                          accumulated.
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
    public OneSidedConvergingDiVectorAccumulator(int dimensions, boolean highIsCritical, double precision,
            int minValuesAccepted, int maxValuesAccepted) {
        super(highIsCritical, precision, minValuesAccepted, maxValuesAccepted);
        accumulatedValue = new DiVector(dimensions);
    }

    /**
     * Compute the "high-low sum" for the given DiVector.
     *
     * @param result A new result DiVector computed by a Random Cut Tree.
     * @return the "high-low sum" for the given DiVector.
     */
    @Override
    protected double getConvergingValue(DiVector result) {
        return result.getHighLowSum();
    }

    /**
     * Add the DiVector to the aggregate DiVector in this accumulator.
     *
     * @param result The new result to add to the accumulated value.
     */
    @Override
    protected void accumulateValue(DiVector result) {
        DiVector.addToLeft(accumulatedValue, result);
    }
}
