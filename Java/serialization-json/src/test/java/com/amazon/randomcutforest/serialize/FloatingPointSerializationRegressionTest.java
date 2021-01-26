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

package com.amazon.randomcutforest.serialize;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.google.gson.Gson;

/**
 * Validate that floating-point numbers are being serialized to JSON in a
 * lossless way.
 */
public class FloatingPointSerializationRegressionTest {
    private RandomCutForestSerDe serDe;

    @BeforeEach
    public void setUp() {
        serDe = new RandomCutForestSerDe();
    }

    @Test
    public void testDoublePrecisionRoundTrip() {
        Random random = new Random();
        double[] array = new double[100];
        for (int i = 0; i < array.length; i++) {
            array[i] = random.nextDouble() * random.nextLong();
        }

        Gson gson = serDe.getGson();
        String json = gson.toJson(array);
        double[] array2 = gson.fromJson(json, double[].class);

        assertArrayEquals(array, array2);
    }

    @Test
    public void testSinglePrecisionRoundTrip() {
        Random random = new Random();
        float[] array = new float[100];
        for (int i = 0; i < array.length; i++) {
            array[i] = random.nextFloat() * random.nextInt();
        }

        Gson gson = serDe.getGson();
        String json = gson.toJson(array);
        float[] array2 = gson.fromJson(json, float[].class);

        assertArrayEquals(array, array2);
    }
}
