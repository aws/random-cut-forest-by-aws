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

import com.amazon.randomcutforest.store.SmallLeafStore;

public class LeafStoreMapperTest {
    private SmallLeafStoreMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new SmallLeafStoreMapper();
    }

    @Test
    public void testRoundTrip() {
        SmallLeafStore store = new SmallLeafStore((short) 10);
        int index1 = store.addLeaf(1, 2, 1);
        int index2 = store.addLeaf(1, 3, 2);
        int index3 = store.addLeaf(4, 5, 1);
        int index4 = store.addLeaf(4, 6, 3);

        SmallLeafStore store2 = mapper.toModel(mapper.toState(store));
        assertEquals(store.getCapacity(), store2.getCapacity());
        assertEquals(store.size(), store2.size());

        IntStream.of(index1, index2, index3, index4).forEach(i -> {
            assertEquals(store.getParent(i), store2.getParent(i));
            assertEquals(store.getPointIndex(i), store2.getPointIndex(i));
            assertEquals(store.getMass(i), store2.getMass(i));
        });
    }
}
