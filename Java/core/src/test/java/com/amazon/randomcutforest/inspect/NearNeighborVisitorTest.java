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

package com.amazon.randomcutforest.inspect;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.tree.INodeView;
import com.amazon.randomcutforest.tree.NodeView;

public class NearNeighborVisitorTest {

    private double[] queryPoint;
    private double distanceThreshold;
    private NearNeighborVisitor visitor;

    @BeforeEach
    public void setUp() {
        queryPoint = new double[] { 7.7, 8.8, -6.6 };
        distanceThreshold = 10.0;
        visitor = new NearNeighborVisitor(queryPoint, distanceThreshold);
    }

    @Test
    public void acceptLeafNear() {
        double[] leafPoint = new double[] { 8.8, 9.9, -5.5 };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(Arrays.copyOf(leafPoint, leafPoint.length));
        when(leafNode.getLiftedLeafPoint()).thenReturn(Arrays.copyOf(leafPoint, leafPoint.length));
        HashMap<Long, Integer> sequenceIndexes = new HashMap<>();
        sequenceIndexes.put(1234L, 1);
        sequenceIndexes.put(5678L, 1);
        when(leafNode.getSequenceIndexes()).thenReturn(sequenceIndexes);

        int depth = 12;
        visitor.acceptLeaf(leafNode, depth);

        Optional<Neighbor> optional = visitor.getResult();
        assertTrue(optional.isPresent());

        Neighbor neighbor = optional.get();
        assertNotSame(leafPoint, neighbor.point);
        assertArrayEquals(leafPoint, neighbor.point);
        assertEquals(Math.sqrt(3 * 1.1 * 1.1), neighbor.distance, EPSILON);
        assertNotSame(leafNode.getSequenceIndexes(), neighbor.sequenceIndexes);
    }

    @Test
    public void acceptLeafNearTimestampsDisabled() {
        double[] leafPoint = new double[] { 8.8, 9.9, -5.5 };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLiftedLeafPoint()).thenReturn(Arrays.copyOf(leafPoint, leafPoint.length));
        when(leafNode.getLeafPoint()).thenReturn(Arrays.copyOf(leafPoint, leafPoint.length));
        assertEquals(0, leafNode.getSequenceIndexes().size());
        int depth = 12;
        visitor.acceptLeaf(leafNode, depth);

        Optional<Neighbor> optional = visitor.getResult();
        assertTrue(optional.isPresent());

        Neighbor neighbor = optional.get();
        assertNotSame(leafPoint, neighbor.point);
        assertArrayEquals(leafPoint, neighbor.point);
        assertEquals(Math.sqrt(3 * 1.1 * 1.1), neighbor.distance, EPSILON);
        assertTrue(neighbor.sequenceIndexes.isEmpty());
    }

    @Test
    public void acceptLeafNotNear() {
        double[] leafPoint = new double[] { 108.8, 209.9, -305.5 };
        INodeView leafNode = mock(NodeView.class);

        HashMap<Long, Integer> sequenceIndexes = new HashMap<>();
        sequenceIndexes.put(1234L, 1);
        sequenceIndexes.put(5678L, 1);
        when(leafNode.getLeafPoint()).thenReturn(leafPoint);
        when(leafNode.getLiftedLeafPoint()).thenReturn(leafPoint);
        when(leafNode.getSequenceIndexes()).thenReturn(sequenceIndexes);

        int depth = 12;
        visitor.acceptLeaf(leafNode, depth);

        Optional<Neighbor> optional = visitor.getResult();
        assertFalse(optional.isPresent());
    }
}
