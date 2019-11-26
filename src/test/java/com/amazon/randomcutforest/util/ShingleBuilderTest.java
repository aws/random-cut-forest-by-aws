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

package com.amazon.randomcutforest.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ShingleBuilderTest {

    private int dimensions;
    private int shingleSize;
    private ShingleBuilder builder;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        shingleSize = 3;
        builder = new ShingleBuilder(dimensions, shingleSize);
    }

    @Test
    public void testNew() {
        assertEquals(dimensions, builder.getInputPointSize());
        assertEquals(dimensions * shingleSize, builder.getShingledPointSize());
        assertFalse(builder.isCyclic());
    }

    @Test
    public void testNewWithInvalidArguments() {
        assertThrows(IllegalArgumentException.class, () -> new ShingleBuilder(0, shingleSize));
        assertThrows(IllegalArgumentException.class, () -> new ShingleBuilder(dimensions, 0));
    }

    @Test
    public void testAddPoint() {
        double[] shingle = builder.getShingle();
        assertArrayEquals(new double[] {0, 0, 0, 0, 0, 0}, shingle);

        builder.addPoint(new double[] {9, 10});
        shingle = builder.getShingle();
        assertArrayEquals(new double[] {0, 0, 0, 0, 9, 10}, shingle);

        builder.addPoint(new double[] {7, 8});
        shingle = builder.getShingle();
        assertArrayEquals(new double[] {0, 0, 9, 10, 7, 8}, shingle);

        builder.addPoint(new double[] {5, 6});
        shingle = builder.getShingle();
        assertArrayEquals(new double[] {9, 10, 7, 8, 5, 6}, shingle);

        builder.addPoint(new double[] {3, 4});
        shingle = builder.getShingle();
        assertArrayEquals(new double[]{7, 8, 5, 6, 3, 4}, shingle);
    }

    @Test
    public void testAddPointCyclic() {
        builder = new ShingleBuilder(dimensions, shingleSize, true);
        double[] shingle = builder.getShingle();
        assertArrayEquals(new double[] {0, 0, 0, 0, 0, 0}, shingle);

        builder.addPoint(new double[] {9, 10});
        shingle = builder.getShingle();
        assertArrayEquals(new double[]{9, 10, 0, 0, 0, 0}, shingle);

        builder.addPoint(new double[] {7, 8});
        shingle = builder.getShingle();
        assertArrayEquals(new double[]{9, 10, 7, 8, 0, 0}, shingle);

        builder.addPoint(new double[] {5, 6});
        shingle = builder.getShingle();
        assertArrayEquals(new double[]{9, 10, 7, 8, 5, 6}, shingle);

        builder.addPoint(new double[] {3, 4});
        shingle = builder.getShingle();
        assertArrayEquals(new double[]{3, 4, 7, 8, 5, 6}, shingle);
    }

    @Test
    public void testAddPointWithInvalidArguments() {
        assertThrows(NullPointerException.class, () -> builder.addPoint(null));

        double[] point = new double[9]; // wrong size of array
        assertThrows(IllegalArgumentException.class, () -> builder.addPoint(point));
    }


    @Test
    public void testShingleCopy() {
        double[] buffer = new double[dimensions * shingleSize];

        builder.addPoint(new double[] {2, 1});
        builder.addPoint(new double[] {4, 3});
        builder.addPoint(new double[] {6, 5});

        double[] shingle = builder.getShingle();
        assertArrayEquals(new double[] {2, 1, 4, 3, 6, 5}, shingle);
        assertArrayEquals(new double[] {0, 0, 0, 0, 0, 0}, buffer);

        builder.getShingle(buffer);
        assertArrayEquals(shingle, buffer);

        buffer[0] = 0;
        assertEquals(2, shingle[0]);
    }

    @Test
    public void testGetShingleWithInvalidArguments() {
        assertThrows(NullPointerException.class, () -> builder.getShingle(null));

        double[] buffer = new double[2]; // wrong size of array
        assertThrows(IllegalArgumentException.class, () -> builder.getShingle(buffer));
    }
}
