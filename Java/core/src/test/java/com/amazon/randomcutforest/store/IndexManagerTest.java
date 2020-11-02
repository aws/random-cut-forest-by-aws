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

public class IndexManagerTest {
    private int capacity;
    private IndexManager store;

    @BeforeEach
    public void setUp() {
        capacity = 4;
        store = new IndexManager(capacity);
    }

    @Test
    public void testNew() {
        assertEquals(capacity, store.getCapacity());
        assertEquals(0, store.size());
    }

    @Test
    public void testNewFromState() {
        int[] freeIndexes = { 4, 0, 3, 1, 2 };
        int freeIndexPointer = 2;
        IndexManager manager = new IndexManager(freeIndexes, freeIndexPointer);
        assertEquals(freeIndexes.length, manager.getCapacity());
        assertEquals(2, manager.size());
        assertEquals(3, manager.takeIndex());
        assertEquals(0, manager.takeIndex());
        assertEquals(4, manager.takeIndex());
    }

    @Test
    public void testTakeIndex() {
        Set<Integer> indexes = new HashSet<>();
        for (int i = 0; i < capacity; i++) {
            indexes.add(store.takeIndex());
            assertEquals(i + 1, store.size());
        }

        // should have returned index 0, 1, 2, and 3
        assertIterableEquals(Arrays.asList(0, 1, 2, 3), indexes.stream().sorted().collect(Collectors.toList()));

        // store is full
        assertThrows(IllegalStateException.class, () -> store.takeIndex());
    }

    @Test
    public void testReleaseIndex() {
        int index1 = store.takeIndex();
        int index2 = store.takeIndex();

        assertEquals(2, store.size());

        store.releaseIndex(index1);
        assertEquals(1, store.size());
    }

    @Test
    public void testCheckValidIndex() {
        int index1 = store.takeIndex();
        int index2 = store.takeIndex();

        // these calls should succeed because the indexes are occupied
        store.checkValidIndex(index1);
        store.checkValidIndex(index2);

        store.releaseIndex(index1);
        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(index1));

        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(-1));
        assertThrows(IllegalArgumentException.class, () -> store.checkValidIndex(capacity));
    }
}
