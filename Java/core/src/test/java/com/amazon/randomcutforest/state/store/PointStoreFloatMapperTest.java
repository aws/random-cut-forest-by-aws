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

package com.amazon.randomcutforest.state.store;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.store.PointStoreFloat;

public class PointStoreFloatMapperTest {
    private PointStoreFloatMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new PointStoreFloatMapper();
    }

    @Test
    public void testRoundTrip() {
        int dimensions = 2;
        int capacity = 4;
        PointStoreFloat store = new PointStoreFloat(dimensions, capacity);

        double[] point1 = { 1.1, -22.2 };
        int index1 = store.add(point1);
        double[] point2 = { 3.3, -4.4 };
        int index2 = store.add(point2);
        double[] point3 = { 10.1, 100.1 };
        int index3 = store.add(point3);

        PointStoreFloat store2 = mapper.toModel(mapper.toState(store));

        assertEquals(capacity, store2.getCapacity());
        assertEquals(3, store2.size());
        assertEquals(dimensions, store2.getDimensions());
        assertArrayEquals(store.getStore(), store2.getStore());
    }
}
