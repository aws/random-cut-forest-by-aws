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

package com.amazon.randomcutforest.state.tree;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.stream.IntStream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.tree.CompactNodeManager;

public class CompactNodeManagerMapperTest {
    private CompactNodeManagerMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new CompactNodeManagerMapper();
    }

    @Test
    public void testRoundTrip() {
        CompactNodeManager manager = new CompactNodeManager(10);

        int nodeIndex1 = manager.addNode(1, 2, 3, 0, 1.0, 1);
        int nodeIndex2 = manager.addNode(1, 4, 5, 1, -1.0, 1);
        int nodeIndex3 = manager.addNode(6, 7, 8, 3, 2.0, 3);
        int nodeIndex4 = manager.addNode(6, 9, 10, 2, -1.0, 1);

        int leafIndex1 = manager.addLeaf(2, 5, 3);
        int leafIndex2 = manager.addLeaf(3, 40, 1);
        int leafIndex3 = manager.addLeaf(10, 2, 2);

        CompactNodeManager manager2 = mapper.toModel(mapper.toState(manager));
        assertEquals(manager.getCapacity(), manager2.getCapacity());

        IntStream.of(nodeIndex1, nodeIndex2, nodeIndex3, nodeIndex4).forEach(i -> {
            assertFalse(manager2.isLeaf(i));
            assertEquals(manager.getParent(i), manager2.getParent(i));
            assertEquals(manager.getLeftChild(i), manager2.getLeftChild(i));
            assertEquals(manager.getRightChild(i), manager2.getRightChild(i));
            assertEquals(manager.getCutDimension(i), manager2.getCutDimension(i));
            assertEquals(manager.getCutValue(i), manager2.getCutValue(i));
            assertEquals(manager.getMass(i), manager2.getMass(i));
        });

        IntStream.of(leafIndex1, leafIndex2, leafIndex3).forEach(i -> {
            assertTrue(manager2.isLeaf(i));
            assertEquals(manager.getParent(i), manager2.getParent(i));
            assertEquals(manager.getPointIndex(i), manager2.getPointIndex(i));
            assertEquals(manager.getMass(i), manager2.getMass(i));
        });
    }
}
