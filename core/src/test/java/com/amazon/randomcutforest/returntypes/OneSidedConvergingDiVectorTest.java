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

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class OneSidedConvergingDiVectorTest {

    private boolean highIsCritical;
    private double precision;
    private int minValuesAccepted;
    private int maxValuesAccepted;
    private int dimensions;
    private OneSidedConvergingDiVectorAccumulator accumulator;

    @BeforeEach
    public void setUp() {
        highIsCritical = true;
        precision = 0.1;
        minValuesAccepted = 5;
        maxValuesAccepted = 100;
        dimensions = 2;
        accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions, highIsCritical, precision,
                minValuesAccepted, maxValuesAccepted);
    }

    @Test
    public void testGetConvergingValue() {
        DiVector vector = new DiVector(dimensions);
        vector.high[0] = 1.1;
        vector.low[1] = 2.3;
        vector.high[1] = 9.6;

        assertEquals(1.1 + 2.3 + 9.6, accumulator.getConvergingValue(vector), EPSILON);
    }

    @Test
    public void testAccumulateValue() {
        DiVector vector1 = new DiVector(dimensions);
        vector1.high[0] = 1.1;
        vector1.low[1] = 2.3;
        vector1.high[1] = 9.6;

        accumulator.accept(vector1);
        DiVector result = accumulator.getAccumulatedValue();
        assertArrayEquals(vector1.high, result.high, EPSILON);
        assertArrayEquals(vector1.low, result.low, EPSILON);

        DiVector vector2 = new DiVector(dimensions);
        vector2.high[0] = 1.1;
        vector2.low[1] = 2.3;
        vector2.high[1] = 9.6;

        accumulator.accept(vector2);
        result = accumulator.getAccumulatedValue();
        DiVector.addToLeft(vector1, vector2);
        assertArrayEquals(vector1.high, result.high, EPSILON);
        assertArrayEquals(vector1.low, result.low, EPSILON);
    }
}
