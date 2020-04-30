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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * This test doubles as a test of the abstract OneSidedStdDevAccumulator class
 */
public class OneSidedConvergingDoubleAccumulatorTest {

	private boolean highIsCritical;
	private double precision;
	private int minValuesAccepted;
	private int maxValuesAccepted;
	private OneSidedConvergingDoubleAccumulator accumulator;

	@BeforeEach
	public void setUp() {
		highIsCritical = true;
		precision = 0.1;
		minValuesAccepted = 5;
		maxValuesAccepted = 100;
		accumulator = new OneSidedConvergingDoubleAccumulator(highIsCritical, precision, minValuesAccepted,
				maxValuesAccepted);
	}

	@Test
	public void testGetConvergingValue() {
		assertEquals(1.23, accumulator.getConvergingValue(1.23));
		assertEquals(-1001.1001, accumulator.getConvergingValue(-1001.1001));
	}

	@Test
	public void testAccumulateValue() {
		double sum = 0.0;
		for (int i = 0; i < 10; i++) {
			double value = Math.random();
			accumulator.accept(value);
			sum += value;
			assertEquals(sum, accumulator.getAccumulatedValue());
		}
	}

	@Test
	public void testConvergenceHighIsCritical() {
		accumulator.accept(0.0);
		accumulator.accept(10.0);
		accumulator.accept(0.0);
		accumulator.accept(10.0);

		// less than minValuesAccepted
		assertEquals(4, accumulator.getValuesAccepted());
		assertFalse(accumulator.isConverged());
		double expectedSum = 20.0;
		assertEquals(expectedSum, accumulator.getAccumulatedValue());

		// each high value should result in a witness to convergence
		// we need 1.0 / precision witnesses in order to converge

		for (int i = 0; i < 1.0 / precision - 1; i++) {
			accumulator.accept(0.0);
			accumulator.accept(10.0);
			assertEquals(6 + 2 * i, accumulator.getValuesAccepted());
			assertFalse(accumulator.isConverged());
			expectedSum += 10.0;
			assertEquals(expectedSum, accumulator.getAccumulatedValue());
		}

		accumulator.accept(0.0);
		assertFalse(accumulator.isConverged());

		// the last required high value
		accumulator.accept(10.0);
		assertTrue(accumulator.isConverged());

		expectedSum += 10.0;
		assertEquals(expectedSum, accumulator.getAccumulatedValue());
	}

	@Test
	public void testConvergenceLowIsCritical() {
		highIsCritical = false;
		accumulator = new OneSidedConvergingDoubleAccumulator(highIsCritical, precision, minValuesAccepted,
				maxValuesAccepted);

		accumulator.accept(0.0);
		accumulator.accept(10.0);
		accumulator.accept(0.0);
		accumulator.accept(10.0);

		// less than minValuesAccepted
		assertFalse(accumulator.isConverged());
		double expectedSum = 20.0;
		assertEquals(expectedSum, accumulator.getAccumulatedValue());

		// each high value should result in a witness to convergence
		// we need 1.0 / precision witnesses in order to converge

		for (int i = 0; i < 1.0 / precision - 1; i++) {
			accumulator.accept(0.0);
			accumulator.accept(10.0);
			assertFalse(accumulator.isConverged());
			expectedSum += 10.0;
			assertEquals(expectedSum, accumulator.getAccumulatedValue());
		}

		accumulator.accept(10.0);
		assertFalse(accumulator.isConverged());

		// the last required low value
		accumulator.accept(0.0);
		assertTrue(accumulator.isConverged());

		expectedSum += 10.0;
		assertEquals(expectedSum, accumulator.getAccumulatedValue());
	}
}
