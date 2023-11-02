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

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArrayNullable;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArrayNullable;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

public class ArrayUtilsTest {

    ArrayUtils utils = new ArrayUtils();

    @ParameterizedTest
    @CsvSource({ "-0.0,0.0", "0.0,0.0", "-0.0:0.0:1.0,0.0:0.0:1.0" })
    public void cleanCopy(String input, String expected) {
        double[] inputArray = array(input);
        double[] cleanCopy = ArrayUtils.cleanCopy(inputArray);
        assertNotSame(inputArray, cleanCopy);
        assertArrayEquals(array(expected), cleanCopy);
    }

    private double[] array(String arrayString) {
        return Arrays.stream(arrayString.split(":")).mapToDouble(Double::valueOf).toArray();
    }

    @Test
    void testNullable() {
        assertNull(toDoubleArrayNullable(null));
        assertNull(toFloatArrayNullable(null));
        float random = new Random().nextFloat();
        assertArrayEquals(toFloatArrayNullable(new double[] { random }), toFloatArray(new double[] { random }));
        assertArrayEquals(toDoubleArrayNullable(new float[] { random }), toDoubleArray(new float[] { random }));
        assertThrows(NullPointerException.class, () -> toDoubleArray(null));
        assertThrows(NullPointerException.class, () -> toFloatArray(null));
    }
}
