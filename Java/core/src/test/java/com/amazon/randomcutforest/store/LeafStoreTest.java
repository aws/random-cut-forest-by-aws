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
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class LeafStoreTest {

    private short capacity;
    private SmallLeafStore store;

    @BeforeEach
    public void setUp() {
        capacity = 3;
        store = new SmallLeafStore(capacity);
    }

    @Test
    public void testNew() {
        assertEquals(capacity, store.getCapacity());
        assertEquals(0, store.size());
    }

    @Test
    public void testAddLeaf() {
        int mass1 = 1;
        short parentIndex1 = 1;
        int pointIndex1 = 2;

        int index1 = store.addLeaf(parentIndex1, pointIndex1, mass1);
        assertEquals(1, store.size());
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(pointIndex1, store.pointIndex[index1]);

        int mass2 = 11;
        short parentIndex2 = 11;
        int pointIndex2 = 12;

        int index2 = store.addLeaf(parentIndex2, pointIndex2, mass2);
        assertEquals(2, store.size());
        assertEquals(mass2, store.mass[index2]);
        assertEquals(parentIndex2, store.parentIndex[index2]);
        assertEquals(pointIndex2, store.pointIndex[index2]);

        // validate that previous values did not change
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(pointIndex1, store.pointIndex[index1]);
    }

    @Test
    public void testAddLeafWhenFull() {
        int mass1 = 1;
        short parentIndex1 = 1;
        int pointIndex1 = 2;

        for (int i = 0; i < capacity; i++) {
            store.addLeaf(parentIndex1, pointIndex1, mass1);
        }

        assertThrows(IllegalStateException.class, () -> store.addLeaf(parentIndex1, pointIndex1, mass1));
    }

    @Test
    public void testRemove() {
        int mass1 = 1;
        short parentIndex1 = 1;
        int pointIndex1 = 2;

        int index1 = store.addLeaf(parentIndex1, pointIndex1, mass1);

        int mass2 = 11;
        short parentIndex2 = 11;
        int pointIndex2 = 12;

        int index2 = store.addLeaf((int) parentIndex2, pointIndex2, mass2);

        store.releaseIndex(index1);
        assertEquals(1, store.size());

        // validate that the values at index2 did not change
        assertEquals(mass2, store.mass[index2]);
        assertEquals(parentIndex2, store.parentIndex[index2]);
        assertEquals(pointIndex2, store.pointIndex[index2]);
    }

    @Test
    public void testRemoveTwice() {
        int mass1 = 1;
        short parentIndex1 = 1;
        int pointIndex1 = 2;

        int index1 = store.addLeaf((int) parentIndex1, pointIndex1, mass1);
        store.releaseIndex(index1);

        assertThrows(IllegalArgumentException.class, () -> store.releaseIndex(index1));
    }

    @Test
    public void testRemoveFromEmptyStore() {
        assertThrows(IllegalArgumentException.class, () -> store.releaseIndex((short) 0));
    }

    @Test
    public void testRemoveInvalidIndex() {
        int mass1 = 1;
        short parentIndex1 = 1;
        int pointIndex1 = 2;

        int index1 = store.addLeaf((int) parentIndex1, pointIndex1, mass1);

        assertThrows(IllegalArgumentException.class, () -> store.releaseIndex((short) -1));
    }
}
