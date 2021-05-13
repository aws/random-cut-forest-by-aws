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


package com.amazon.randomcutforest.util;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ArrayPackingTest {
    private Random rng;

    @BeforeEach
    public void setUp() {
        rng = new Random();
    }

    @Test
    public void testLogMax() {
        long[] bases = new long[] {2, 101, 3_456_789};
        Arrays.stream(bases)
                .forEach(
                        base -> {
                            int log = ArrayPacking.logMax(base);
                            assertTrue(Math.pow(base, log + 1) >= Integer.MAX_VALUE);
                            assertTrue(Math.pow(base, log) < Integer.MAX_VALUE);
                        });
    }

    @Test
    public void testLogMaxInvalid() {
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(1));
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(0));
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(-123467890));
    }

    @Test
    public void testIntsPackRoundTrip() {
        int inputLength = 100;
        int[] inputArray = rng.ints().limit(inputLength).toArray();
        assertArrayEquals(
                inputArray, ArrayPacking.unpackInts(ArrayPacking.pack(inputArray, false), false));
        assertArrayEquals(
                inputArray, ArrayPacking.unpackInts(ArrayPacking.pack(inputArray, true), true));
    }

    @Test
    public void testUnpackIntsWithLengthGiven() {
        int inputLength = 100;
        int[] inputArray = rng.ints().limit(inputLength).toArray();

        int[] uncompressed = ArrayPacking.pack(inputArray, false);
        int[] compressed = ArrayPacking.pack(inputArray, true);

        int[] result = ArrayPacking.unpackInts(uncompressed, 50, false);
        assertEquals(50, result.length);
        assertArrayEquals(Arrays.copyOf(inputArray, 50), result);

        result = ArrayPacking.unpackInts(compressed, 50, true);
        assertEquals(50, result.length);
        assertArrayEquals(Arrays.copyOf(inputArray, 50), result);

        result = ArrayPacking.unpackInts(uncompressed, 200, false);
        assertEquals(200, result.length);
        assertArrayEquals(inputArray, Arrays.copyOf(result, 100));
        for (int i = 100; i < 200; i++) {
            assertEquals(0, result[i]);
        }

        result = ArrayPacking.unpackInts(compressed, 200, true);
        assertEquals(200, result.length);
        assertArrayEquals(inputArray, Arrays.copyOf(result, 100));
        for (int i = 100; i < 200; i++) {
            assertEquals(0, result[i]);
        }
    }

    @Test
    public void testPackDoublesRoundTrip() {
        int inputLength = 100;
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        assertArrayEquals(inputArray, ArrayPacking.unpackDoubles(ArrayPacking.pack(inputArray)));
    }

    @Test
    public void testPackFloatsRoundTrip() {
        int inputLength = 100;
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        assertArrayEquals(inputArray, ArrayPacking.unpackFloats(ArrayPacking.pack(inputArray)));
    }

    @Test
    public void testPackDoublesWithLength() {
        int inputLength = 100;
        int packLength = 76;
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        byte[] bytes = ArrayPacking.pack(inputArray, packLength);
        double[] outputArray = ArrayPacking.unpackDoubles(bytes);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
    }

    @Test
    public void testPackFloatsWithLength() {
        int inputLength = 100;
        int packLength = 76;
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        byte[] bytes = ArrayPacking.pack(inputArray, packLength);
        float[] outputArray = ArrayPacking.unpackFloats(bytes);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
    }

    @Test
    public void testUnpackDoublesWithLength() {
        int inputLength = 100;
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        byte[] bytes = ArrayPacking.pack(inputArray);

        int unpackLength1 = 25;
        double[] outputArray1 = ArrayPacking.unpackDoubles(bytes, unpackLength1);
        assertEquals(unpackLength1, outputArray1.length);
        assertArrayEquals(Arrays.copyOf(inputArray, unpackLength1), outputArray1);

        int unpackLength2 = 123;
        double[] outputArray2 = ArrayPacking.unpackDoubles(bytes, unpackLength2);
        assertEquals(unpackLength2, outputArray2.length);
        assertArrayEquals(inputArray, Arrays.copyOf(outputArray2, inputLength));
        for (int i = inputLength; i < unpackLength2; i++) {
            assertEquals(0.0, outputArray2[i]);
        }
    }

    @Test
    public void testUnpackFloatWithLength() {
        int inputLength = 100;
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        byte[] bytes = ArrayPacking.pack(inputArray);

        int unpackLength1 = 25;
        float[] outputArray1 = ArrayPacking.unpackFloats(bytes, unpackLength1);
        assertEquals(unpackLength1, outputArray1.length);
        assertArrayEquals(Arrays.copyOf(inputArray, unpackLength1), outputArray1);

        int unpackLength2 = 123;
        float[] outputArray2 = ArrayPacking.unpackFloats(bytes, unpackLength2);
        assertEquals(unpackLength2, outputArray2.length);
        assertArrayEquals(inputArray, Arrays.copyOf(outputArray2, inputLength));
        for (int i = inputLength; i < unpackLength2; i++) {
            assertEquals(0.0, outputArray2[i]);
        }
    }
}
