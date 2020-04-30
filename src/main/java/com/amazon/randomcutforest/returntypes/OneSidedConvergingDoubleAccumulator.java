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
 * A converging accumulator using a one-sided standard deviation tests. The
 * accumulator tests the submitted values for convergence and returns the sum of
 * all submitted values.
 */
public class OneSidedConvergingDoubleAccumulator extends OneSidedStDevAccumulator<Double> {

	/**
	 * Create a new converging accumulator that uses a one-sided standard deviation
	 * test.
	 *
	 * @param highIsCritical
	 *            Set to 'true' if we care more about high values of the converging
	 *            scalar than low values. Set to 'false' if the opposite is true.
	 * @param precision
	 *            The number of witnesses required before declaring convergence will
	 *            be at least 1.0 / precision.
	 * @param minValuesAccepted
	 *            The user-specified minimum number of values visited before
	 *            returning a result. Note that {@link #isConverged()} may return
	 *            true before accepting this number of results if the
	 * @param maxValuesAccepted
	 *            The maximum number of values that will be accepted by this
	 *            accumulator.
	 */
	public OneSidedConvergingDoubleAccumulator(boolean highIsCritical, double precision, int minValuesAccepted,
			int maxValuesAccepted) {
		super(highIsCritical, precision, minValuesAccepted, maxValuesAccepted);
		accumulatedValue = 0.0;
	}

	/**
	 * We are testing for convergence directly on the submitted double values, hence
	 * we just return the argument as-is.
	 *
	 * @param result
	 *            A new result value computed by a Random Cut Tree.
	 * @return the result value.
	 */
	@Override
	protected double getConvergingValue(Double result) {
		return result;
	}

	/**
	 * Add the result to the sum of result values.
	 *
	 * @param result
	 *            The new result to add to the accumulated value.
	 */
	@Override
	protected void accumulateValue(Double result) {
		accumulatedValue += result;
	}
}
