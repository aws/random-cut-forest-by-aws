/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.Arrays;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

public class DiVectorTest {

    int dimensions;
    private DiVector vector;

    @BeforeEach
    public void setUp() {
        dimensions = 3;
        vector = new DiVector(dimensions);
    }

    @Test
    public void testNew() {
        double[] expected = new double[dimensions];
        assertEquals(dimensions, vector.getDimensions());
        assertArrayEquals(expected, vector.high);
        assertArrayEquals(expected, vector.low);
    }

    @Test
    public void testAddToLeft() {
        DiVector left = new DiVector(dimensions);
        DiVector right = new DiVector(dimensions);
        for (int i = 0; i < dimensions; i++) {
            left.low[i] = Math.random();
            left.high[i] = Math.random();
            right.low[i] = Math.random();
            right.high[i] = Math.random();
        }

        DiVector leftCopy = new DiVector(dimensions);
        System.arraycopy(left.low, 0, leftCopy.low, 0, dimensions);
        System.arraycopy(left.high, 0, leftCopy.high, 0, dimensions);

        DiVector rightCopy = new DiVector(dimensions);
        System.arraycopy(right.low, 0, rightCopy.low, 0, dimensions);
        System.arraycopy(right.high, 0, rightCopy.high, 0, dimensions);

        DiVector result = DiVector.addToLeft(left, right);

        assertSame(result, left);
        assertArrayEquals(rightCopy.low, right.low);
        assertArrayEquals(rightCopy.high, right.high);

        for (int i = 0; i < dimensions; i++) {
            assertEquals(leftCopy.low[i] + right.low[i], left.low[i]);
            assertEquals(leftCopy.high[i] + right.high[i], left.high[i]);
        }
    }

    @Test
    public void testScale() {
        vector.high[0] = 1.1;
        vector.high[2] = 3.1;
        vector.low[1] = 2.2;

        double z = 9.9;
        DiVector result = vector.scale(z);

        double[] expected = new double[] {1.1 * 9.9, 0.0, 3.1 * 9.9};
        assertArrayEquals(expected, result.high);

        expected = new double[] {0.0, 2.2 * 9.9, 0.0};
        assertArrayEquals(expected, result.low);

        DiVector emptyVector = new DiVector(dimensions);
        emptyVector.scale(123.0);
        expected = new double[dimensions];
        assertArrayEquals(expected, emptyVector.low);
        assertArrayEquals(expected, emptyVector.high);
    }

    @Test
    public void testGetHighLowSum() {
        vector.high[2] = 3.1;
        vector.low[1] = 2.2;

        assertEquals(3.1 + 2.2, vector.getHighLowSum());
    }

    @Test
    public void testRenormalize() {
        vector.high[0] = 1.1;
        vector.high[2] = 3.1;
        vector.low[1] = 2.2;

        assertEquals(1.1 + 3.1 + 2.2, vector.getHighLowSum());

        vector.renormalize(100.0);

        assertEquals(100.0, vector.getHighLowSum());
    }

    @Test
    public void testComponentwiseTransform() {
        vector.high[0] = 1.1;
        vector.high[1] = 2.1;
        vector.high[2] = 3.1;
        vector.low[0] = 101.1;
        vector.low[1] = 202.1;
        vector.low[2] = 303.1;

        double[] highCopy = Arrays.copyOf(vector.high, dimensions);
        double[] lowCopy = Arrays.copyOf(vector.low, dimensions);
        vector.componentwiseTransform(x -> 2 * x - 1);

        for (int i = 0; i < dimensions; i++) {
            assertEquals(2 * highCopy[i] - 1, vector.high[i], EPSILON);
            assertEquals(2 * lowCopy[i] - 1, vector.low[i], EPSILON);
        }
    }
}
