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

package com.amazon.randomcutforest.sampler;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.executor.Sequential;

public class WeightedTest {

    @Test
    public void testNew() {
        Sequential<double[]> point = new Sequential(new double[] { 0.99, -55.2 }, 1.23f, 999L);
        assertEquals(1.23f, point.getWeight());
        assertEquals(999L, point.getSequenceIndex());
        assertArrayEquals(new double[] { 0.99, -55.2 }, point.getValue());
    }

}
