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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.stream.IntStream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.store.NodeStore;

public class NodeStoreMapperTest {
    private NodeStoreMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new NodeStoreMapper();
    }

    @Test
    public void testRoundTrip() {
        NodeStore store = new NodeStore((short) 10);
        int index1 = store.addNode(1, 2, 3, 1, 6.6, 1);
        int index2 = store.addNode(1, 4, 5, 2, -14.8, 2);
        int index3 = store.addNode(6, 7, 8, 1, 9.8, 4);
        int index4 = store.addNode(6, 10, 11, 4, -1000.01, 1);

        NodeStore store2 = mapper.toModel(mapper.toState(store));
        assertEquals(store.getCapacity(), store2.getCapacity());
        assertEquals(store.size(), store2.size());

        IntStream.of(index1, index2, index3, index4).forEach(i -> {
            assertEquals(store.getLeftIndex(i), store2.getLeftIndex(i));
            assertEquals(store.getRightIndex(i), store2.getRightIndex(i));
            assertEquals(store.getCutDimension(i), store2.getCutDimension(i));
            assertEquals(store.getCutValue(i), store2.getCutValue(i));
        });
    }
}
