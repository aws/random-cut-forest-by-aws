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
import static org.junit.jupiter.api.Assertions.assertNotSame;

import java.util.Arrays;

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
}
