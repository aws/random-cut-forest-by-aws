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
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class RangeVectorTest {

    int dimensions;
    private RangeVector vector;

    @BeforeEach
    public void setUp() {
        dimensions = 3;
        vector = new RangeVector(dimensions);
    }

    @Test
    public void testNew() {
        assertThrows(IllegalArgumentException.class, () -> new RangeVector(0));
        assertThrows(IllegalArgumentException.class, () -> new RangeVector(new float[0]));
        float[] expected = new float[dimensions];
        assertArrayEquals(expected, vector.values);
        assertArrayEquals(expected, vector.upper);
        assertArrayEquals(expected, vector.lower);

        float[] another = new float[0];
        assertThrows(IllegalArgumentException.class, () -> new RangeVector(another, another, another));
        assertThrows(IllegalArgumentException.class,
                () -> new RangeVector(expected, expected, new float[dimensions + 1]));
        assertThrows(IllegalArgumentException.class,
                () -> new RangeVector(expected, new float[dimensions + 1], expected));
        assertThrows(IllegalArgumentException.class,
                () -> new RangeVector(new float[dimensions + 1], expected, expected));
        assertDoesNotThrow(() -> new RangeVector(expected, expected, expected));

        assertThrows(IllegalArgumentException.class,
                () -> new RangeVector(expected, new float[] { -1f, 0f, 0f }, expected));
        assertDoesNotThrow(() -> new RangeVector(expected, expected, new float[] { -1f, 0f, 0f }));

        assertThrows(IllegalArgumentException.class,
                () -> new RangeVector(expected, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }));
        assertDoesNotThrow(() -> new RangeVector(expected, new float[] { 1f, 0f, 0f }, new float[] { -1f, 0f, 0f }));
    }

    @Test
    public void testScale() {
        vector.upper[0] = 1.1f;
        vector.upper[2] = 3.1f;
        vector.upper[1] = 3.1f;
        vector.lower[1] = -2.2f;

        float z = 9.9f;
        assertThrows(IllegalArgumentException.class, () -> vector.scale(0, -1.0f));
        assertThrows(IllegalArgumentException.class, () -> vector.scale(-1, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> vector.scale(dimensions + 1, 1.0f));
        vector.scale(0, z);

        float[] expected = new float[] { 1.1f * 9.9f, 3.1f, 3.1f };
        assertArrayEquals(expected, vector.upper, 1e-6f);

        expected = new float[] { 0.0f, -2.2f, 0.0f };
        assertArrayEquals(expected, vector.lower);

        vector.scale(1, 2 * z);
        assertArrayEquals(new float[] { 1.1f * 9.9f, 3.1f * 2 * z, 3.1f }, vector.upper, 1e-6f);
        assertArrayEquals(new float[] { 0f, -2.2f * 2 * z, 0f }, vector.lower, 1e-6f);
    }

    @Test
    public void testShift() {
        vector.upper[0] = 1.1f;
        vector.upper[2] = 3.1f;
        vector.lower[1] = -2.2f;

        float z = -9.9f;
        assertThrows(IllegalArgumentException.class, () -> vector.shift(-1, z));
        assertThrows(IllegalArgumentException.class, () -> vector.shift(dimensions + 1, z));
        vector.shift(0, z);

        float[] expected = new float[] { 1.1f - 9.9f, 0.0f, 3.1f };
        assertArrayEquals(expected, vector.upper, 1e-6f);

        expected = new float[] { z, -2.2f, 0.0f };
        assertArrayEquals(expected, vector.lower);

        assertArrayEquals(new float[] { z, 0, 0 }, vector.values, 1e-6f);
    }

}
