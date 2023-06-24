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

package com.amazon.randomcutforest.parkservices.returntypes;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.RangeVector;

public class TimedRangeVectorTest {

    int dimensions;
    int horizon;
    private TimedRangeVector vector;

    @BeforeEach
    public void setUp() {
        dimensions = 4;
        horizon = 2;
        vector = new TimedRangeVector(dimensions, horizon);
    }

    @Test
    public void testNew() {
        assertThrows(IllegalArgumentException.class, () -> new TimedRangeVector(2, -2));
        assertThrows(IllegalArgumentException.class, () -> new TimedRangeVector(-2, 2));
        assertThrows(IllegalArgumentException.class, () -> new TimedRangeVector(5, 2));
        assertDoesNotThrow(() -> new TimedRangeVector(6, 2));
        assertThrows(IllegalArgumentException.class, () -> new TimedRangeVector(new RangeVector(8), 3));
        assertDoesNotThrow(() -> new TimedRangeVector(new RangeVector(9), 3));

        assertThrows(IllegalArgumentException.class,
                () -> new TimedRangeVector(new RangeVector(5), new long[2], new long[2], new long[2]));
        assertThrows(IllegalArgumentException.class,
                () -> new TimedRangeVector(new RangeVector(4), new long[2], new long[2], new long[1]));
        assertThrows(IllegalArgumentException.class,
                () -> new TimedRangeVector(new RangeVector(4), new long[2], new long[1], new long[1]));
    }

    @Test
    public void testScale() {
        assertTrue(vector.timeStamps.length == 2);
        vector.timeStamps[0] = 100L;
        vector.upperTimeStamps[0] = 120L;
        vector.lowerTimeStamps[0] = -82L;
        vector.lowerTimeStamps[1] = -100L;
        assertThrows(IllegalArgumentException.class, () -> vector.scaleTime(-1, 1.0));
        assertThrows(IllegalArgumentException.class, () -> vector.scaleTime(3, 1.0));
        assertThrows(IllegalArgumentException.class, () -> vector.scaleTime(0, -1.0));

        vector.scaleTime(0, 0.5);
        assertArrayEquals(vector.timeStamps, new long[] { 50, 0 });
        assertArrayEquals(vector.upperTimeStamps, new long[] { 60, 0 });
        assertArrayEquals(vector.lowerTimeStamps, new long[] { -41, -100 });
    }

    @Test
    public void testShift() {
        vector.timeStamps[0] = 100L;
        vector.upperTimeStamps[0] = 120L;
        vector.lowerTimeStamps[0] = -82L;
        vector.lowerTimeStamps[1] = -100L;
        assertThrows(IllegalArgumentException.class, () -> vector.shiftTime(-1, 1L));
        assertThrows(IllegalArgumentException.class, () -> vector.shiftTime(3, 1L));

        vector.shiftTime(1, 13);

        TimedRangeVector newVector = new TimedRangeVector(vector);
        assertArrayEquals(newVector.timeStamps, new long[] { 100, 13 });
        assertArrayEquals(newVector.upperTimeStamps, new long[] { 120, 13 });
        assertArrayEquals(newVector.lowerTimeStamps, new long[] { -82, -87 });

        newVector.shiftTime(1, -130);
        assertArrayEquals(vector.timeStamps, new long[] { 100, 13 });
        assertArrayEquals(vector.upperTimeStamps, new long[] { 120, 13 });
        assertArrayEquals(vector.lowerTimeStamps, new long[] { -82, -87 });

        assertThrows(IllegalArgumentException.class,
                () -> new TimedRangeVector(new RangeVector(4), newVector.timeStamps, new long[2], new long[2]));

        assertThrows(IllegalArgumentException.class, () -> new TimedRangeVector(new RangeVector(4),
                newVector.timeStamps, new long[] { 101L, 0L }, new long[2]));

        TimedRangeVector another = new TimedRangeVector(new RangeVector(4), newVector.timeStamps,
                new long[] { 101L, 0L }, newVector.lowerTimeStamps);
        assertArrayEquals(another.timeStamps, new long[] { 100, -117 });
        assertArrayEquals(another.upperTimeStamps, new long[] { 101, 0 });
        assertArrayEquals(another.lowerTimeStamps, new long[] { -82, -217 });
    }

}
