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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class InterpolationMeasureTest {

    private int dimensions;
    private int sampleSize;
    private InterpolationMeasure output;

    @BeforeEach
    public void setUp() {
        dimensions = 3;
        sampleSize = 99;
        output = new InterpolationMeasure(dimensions, sampleSize);
    }

    @Test
    public void testNew() {
        double[] zero = new double[3];
        assertArrayEquals(zero, output.measure.high);
        assertArrayEquals(zero, output.distances.high);
        assertArrayEquals(zero, output.probMass.high);
        assertArrayEquals(zero, output.measure.low);
        assertArrayEquals(zero, output.distances.low);
        assertArrayEquals(zero, output.probMass.low);
    }

    @Test
    public void testAddToLeft() {
        InterpolationMeasure other1 = new InterpolationMeasure(dimensions, sampleSize);
        InterpolationMeasure other2 = new InterpolationMeasure(dimensions, sampleSize);

        for (int i = 0; i < dimensions; i++) {
            output.probMass.high[i] = 2 * i;
            output.probMass.low[i] = 2 * i + 1;
            output.distances.high[i] = 4 * i;
            output.distances.low[i] = 4 * i + 2;
            output.measure.high[i] = 6 * i;
            output.measure.low[i] = 6 * i + 3;

            other1.probMass.high[i] = other2.probMass.high[i] = 8 * i;
            other1.distances.high[i] = other2.distances.high[i] = 10 * i;
            other1.measure.high[i] = other2.measure.high[i] = 12 * i;

            other1.probMass.low[i] = other2.probMass.low[i] = 8 * i + 4;
            other1.distances.low[i] = other2.distances.low[i] = 10 * i + 5;
            other1.measure.low[i] = other2.measure.low[i] = 12 * i + 6;
        }

        assertArrayEquals(other1.probMass.high, other2.probMass.high);
        assertArrayEquals(other1.distances.high, other2.distances.high);
        assertArrayEquals(other1.measure.high, other2.measure.high);
        assertArrayEquals(other1.probMass.low, other2.probMass.low);
        assertArrayEquals(other1.distances.low, other2.distances.low);
        assertArrayEquals(other1.measure.low, other2.measure.low);

        InterpolationMeasure.addToLeft(output, other1);

        for (int i = 0; i < dimensions; i++) {
            assertEquals(2 * i + 8 * i, output.probMass.high[i]);
            assertEquals(4 * i + 10 * i, output.distances.high[i]);
            assertEquals(6 * i + 12 * i, output.measure.high[i]);
            assertEquals(2 * i + 8 * i + 5, output.probMass.low[i]);
            assertEquals(4 * i + 10 * i + 7, output.distances.low[i]);
            assertEquals(6 * i + 12 * i + 9, output.measure.low[i]);
        }

        assertArrayEquals(other1.probMass.high, other2.probMass.high);
        assertArrayEquals(other1.distances.high, other2.distances.high);
        assertArrayEquals(other1.measure.high, other2.measure.high);
        assertArrayEquals(other1.probMass.low, other2.probMass.low);
        assertArrayEquals(other1.distances.low, other2.distances.low);
        assertArrayEquals(other1.measure.low, other2.measure.low);
    }

    @Test
    public void testScale() {
        InterpolationMeasure copy = new InterpolationMeasure(dimensions, sampleSize);

        for (int i = 0; i < dimensions; i++) {
            output.probMass.high[i] = copy.probMass.high[i] = 2 * i;
            output.distances.high[i] = copy.distances.high[i] = 4 * i;
            output.measure.high[i] = copy.measure.high[i] = 6 * i;
            output.probMass.low[i] = copy.probMass.low[i] = 2 * i + 1;
            output.distances.low[i] = copy.distances.low[i] = 4 * i + 2;
            output.measure.low[i] = copy.measure.low[i] = 6 * i + 3;
        }

        assertArrayEquals(copy.probMass.high, output.probMass.high);
        assertArrayEquals(copy.distances.high, output.distances.high);
        assertArrayEquals(copy.measure.high, output.measure.high);
        assertArrayEquals(copy.probMass.low, output.probMass.low);
        assertArrayEquals(copy.distances.low, output.distances.low);
        assertArrayEquals(copy.measure.low, output.measure.low);

        InterpolationMeasure result = output.scale(0.9);

        assertArrayEquals(copy.probMass.low, output.probMass.low);
        assertArrayEquals(copy.distances.low, output.distances.low);
        assertArrayEquals(copy.measure.low, output.measure.low);
        assertArrayEquals(copy.probMass.high, output.probMass.high);
        assertArrayEquals(copy.distances.high, output.distances.high);
        assertArrayEquals(copy.measure.high, output.measure.high);

        for (int i = 0; i < dimensions; i++) {
            assertEquals(2 * i * 0.9, result.probMass.high[i]);
            assertEquals(4 * i * 0.9, result.distances.high[i]);
            assertEquals(6 * i * 0.9, result.measure.high[i]);
            assertEquals((2 * i + 1) * 0.9, result.probMass.low[i]);
            assertEquals((4 * i + 2) * 0.9, result.distances.low[i]);
            assertEquals((6 * i + 3) * 0.9, result.measure.low[i]);
        }
    }
}
