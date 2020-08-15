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

package com.amazon.randomcutforest.store;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class SmallStoreCommonTest {
    private short capacity;
    private SmallStoreCommon store;

    @BeforeEach
    public void setUp() {
        capacity = 4;
        store = new SmallStoreCommon(capacity);
    }

    @Test
    public void testNew() {
        assertEquals(capacity, store.getCapacity());
        assertEquals(0, store.size());
    }

    @Test
    public void testTakeIndex() {
        Set<Short> indexes = new HashSet<>();
        for (int i = 0; i < capacity; i++) {
            indexes.add(store.takeIndex());
            assertEquals(i + 1, store.size());
        }

        // should have returned index 0, 1, 2, and 3
        assertIterableEquals(Arrays.asList((short) 0, (short) 1, (short) 2, (short) 3),
                indexes.stream().sorted().collect(Collectors.toList()));

        // store is full
        assertThrows(IllegalStateException.class, () -> store.takeIndex());
    }

    @Test
    public void testReleaseIndex() {
        short index1 = store.takeIndex();
        short index2 = store.takeIndex();

        assertEquals(2, store.size());

        store.releaseIndex(index1);
        assertEquals(1, store.size());
    }

    @Test
    public void testCheckValidIndex() {
        short index1 = store.takeIndex();
        short index2 = store.takeIndex();

        // these calls should succeed because the indexes are occupied
        store.checkValidIndex(index1);
        store.checkValidIndex(index2);

        store.releaseIndex(index1);
        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(index1));

        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(-1));
        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(capacity));
    }
}
