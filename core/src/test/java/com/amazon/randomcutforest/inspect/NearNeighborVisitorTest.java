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
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.tree.Node;

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
        Node leafNode = spy(new Node(leafPoint));

        Set<Long> sequenceIndexes = new HashSet<>();
        sequenceIndexes.add(1234L);
        sequenceIndexes.add(5678L);
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
        assertThat(neighbor.sequenceIndexes, containsInAnyOrder(leafNode.getSequenceIndexes().toArray()));
    }

    @Test
    public void acceptLeafNearTimestampsDisabled() {
        double[] leafPoint = new double[] { 8.8, 9.9, -5.5 };
        Node leafNode = new Node(leafPoint);
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
        Node leafNode = spy(new Node(leafPoint));

        Set<Long> sequenceIndexes = new HashSet<>();
        sequenceIndexes.add(1234L);
        sequenceIndexes.add(5678L);
        when(leafNode.getSequenceIndexes()).thenReturn(sequenceIndexes);

        int depth = 12;
        visitor.acceptLeaf(leafNode, depth);

        Optional<Neighbor> optional = visitor.getResult();
        assertFalse(optional.isPresent());
    }
}
