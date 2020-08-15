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

public class NodeStoreTest {

    private short capacity;
    private NodeStore store;

    @BeforeEach
    public void setUp() {
        capacity = 3;
        store = new NodeStore(capacity);
    }

    @Test
    public void testNew() {
        assertEquals(capacity, store.getCapacity());
        assertEquals(0, store.size());
    }

    @Test
    public void testAddNode() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        short index1 = store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1);
        assertEquals(1, store.size());
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(leftIndex1, store.leftIndex[index1]);
        assertEquals(rightIndex1, store.rightIndex[index1]);
        assertEquals(cutDimension1, store.cutDimension[index1]);
        assertEquals(cutValue1, store.cutValue[index1]);

        int mass2 = 11;
        short parentIndex2 = 11;
        short leftIndex2 = 12;
        short rightIndex2 = 13;
        int cutDimension2 = 14;
        float cutValue2 = 15.5f;

        short index2 = store.addNode(parentIndex2, leftIndex2, rightIndex2, cutDimension2, cutValue2, mass2);
        assertEquals(2, store.size());
        assertEquals(mass2, store.mass[index2]);
        assertEquals(parentIndex2, store.parentIndex[index2]);
        assertEquals(leftIndex2, store.leftIndex[index2]);
        assertEquals(rightIndex2, store.rightIndex[index2]);
        assertEquals(cutDimension2, store.cutDimension[index2]);
        assertEquals(cutValue2, store.cutValue[index2]);

        // validate that previous values did not change
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(leftIndex1, store.leftIndex[index1]);
        assertEquals(rightIndex1, store.rightIndex[index1]);
        assertEquals(cutDimension1, store.cutDimension[index1]);
        assertEquals(cutValue1, store.cutValue[index1]);
    }

    @Test
    public void testStageNode() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        short index1 = store.stageNode().parentIndex(parentIndex1).leftIndex(leftIndex1).rightIndex(rightIndex1)
                .cutDimension(cutDimension1).cutValue(cutValue1).mass(mass1).add();

        assertEquals(1, store.size());
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(leftIndex1, store.leftIndex[index1]);
        assertEquals(rightIndex1, store.rightIndex[index1]);
        assertEquals(cutDimension1, store.cutDimension[index1]);
        assertEquals(cutValue1, store.cutValue[index1]);

        int mass2 = 11;
        short parentIndex2 = 11;
        short leftIndex2 = 12;
        short rightIndex2 = 13;
        int cutDimension2 = 14;
        float cutValue2 = 15.5f;

        short index2 = store.stageNode().parentIndex(parentIndex2).leftIndex(leftIndex2).rightIndex(rightIndex2)
                .cutDimension(cutDimension2).cutValue(cutValue2).mass(mass2).add();

        assertEquals(2, store.size());
        assertEquals(mass2, store.mass[index2]);
        assertEquals(parentIndex2, store.parentIndex[index2]);
        assertEquals(leftIndex2, store.leftIndex[index2]);
        assertEquals(rightIndex2, store.rightIndex[index2]);
        assertEquals(cutDimension2, store.cutDimension[index2]);
        assertEquals(cutValue2, store.cutValue[index2]);

        // validate that previous values did not change
        assertEquals(mass1, store.mass[index1]);
        assertEquals(parentIndex1, store.parentIndex[index1]);
        assertEquals(leftIndex1, store.leftIndex[index1]);
        assertEquals(rightIndex1, store.rightIndex[index1]);
        assertEquals(cutDimension1, store.cutDimension[index1]);
        assertEquals(cutValue1, store.cutValue[index1]);
    }

    @Test
    public void testAddNodeWhenFull() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        for (int i = 0; i < capacity; i++) {
            store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1);
        }

        assertThrows(IllegalStateException.class,
                () -> store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1));
    }

    @Test
    public void testRemove() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        short index1 = store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1);

        int mass2 = 11;
        short parentIndex2 = 11;
        short leftIndex2 = 12;
        short rightIndex2 = 13;
        int cutDimension2 = 14;
        float cutValue2 = 15.5f;

        short index2 = store.addNode(parentIndex2, leftIndex2, rightIndex2, cutDimension2, cutValue2, mass2);

        store.removeNode(index1);
        assertEquals(1, store.size());

        // validate that the values at index2 did not change
        assertEquals(mass2, store.mass[index2]);
        assertEquals(parentIndex2, store.parentIndex[index2]);
        assertEquals(leftIndex2, store.leftIndex[index2]);
        assertEquals(rightIndex2, store.rightIndex[index2]);
        assertEquals(cutDimension2, store.cutDimension[index2]);
        assertEquals(cutValue2, store.cutValue[index2]);
    }

    @Test
    public void testRemoveTwice() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        short index1 = store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1);
        store.removeNode(index1);

        assertThrows(IllegalArgumentException.class, () -> store.removeNode(index1));
    }

    @Test
    public void testRemoveFromEmptyStore() {
        assertThrows(IllegalArgumentException.class, () -> store.removeNode((short) 0));
    }

    @Test
    public void testRemoveInvalidIndex() {
        int mass1 = 1;
        short parentIndex1 = 1;
        short leftIndex1 = 2;
        short rightIndex1 = 3;
        int cutDimension1 = 4;
        float cutValue1 = 5.5f;

        short index1 = store.addNode(parentIndex1, leftIndex1, rightIndex1, cutDimension1, cutValue1, mass1);

        assertThrows(IllegalArgumentException.class, () -> store.removeNode((short) -1));
    }
}
