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

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DensityOutputTest {

    private int dimensions;
    private int sampleSize;
    private DensityOutput output;

    @BeforeEach
    public void setUp() {
        dimensions = 3;
        sampleSize = 99;
        output = new DensityOutput(dimensions, sampleSize);
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
        DensityOutput other1 = new DensityOutput(dimensions, sampleSize);
        DensityOutput other2 = new DensityOutput(dimensions, sampleSize);

        for (int i = 0; i < dimensions; i++) {
            output.probMass.high[i] = 2 * i;
            output.probMass.low[i] = 2 * i + 1;
            output.distances.high[i] = 4 * i;
            output.distances.low[i] = 4 * i +2;
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


        DensityOutput.addToLeft(output, other1);

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
    public void testGetDensity() {
        for (int i = 0; i < dimensions; i++) {
            output.probMass.high[i] = 2 * i;
            output.distances.high[i] = 4 * i;
            output.measure.high[i] = 6 * i;
            output.probMass.low[i] = 2 * i + 1;
            output.distances.low[i] = 4 * i + 2;
            output.measure.low[i] = 6 * i + 3;
        }

        double q = 0.5;
        double density = output.getDensity(q,3);
        DiVector densityVector = output.getDirectionalDensity(q,3);

        double sumOfPoints = output.measure.getHighLowSum() / sampleSize;
        double sumOfFactors = 0.0;
        for (int i = 0; i < dimensions; i++) {
            double mass = output.probMass.getHighLowSum(i);
            double distance = output.distances.getHighLowSum(i);
            double t = distance / mass;
            t = Math.pow(t, dimensions) * mass;
            sumOfFactors += t;
        }

        assertEquals(sumOfPoints / (q*sumOfPoints+sumOfFactors), density, EPSILON);

        // for contrib, do not scale sum of points by sample size
        sumOfPoints = output.measure.getHighLowSum();


        for (int i = 0; i < dimensions; i++) {
            assertEquals(output.measure.high[i] * density / sumOfPoints, densityVector.high[i], EPSILON);
            assertEquals(output.measure.low[i] * density / sumOfPoints, densityVector.low[i], EPSILON);
        }

        assertEquals(output.getDensity(DensityOutput.DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, dimensions),
                output.getDensity());

        densityVector = output.getDirectionalDensity(DensityOutput.DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, dimensions);
        DiVector defaultDensityVector = output.getDirectionalDensity();
        for (int i = 0; i < dimensions; i++) {
            assertEquals(densityVector.high[i], defaultDensityVector.high[i], EPSILON);
            assertEquals(densityVector.low[i], defaultDensityVector.low[i], EPSILON);
        }
    }

}
