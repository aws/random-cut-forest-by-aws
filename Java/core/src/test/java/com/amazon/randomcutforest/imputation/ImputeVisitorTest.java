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

package com.amazon.randomcutforest.imputation;

import static com.amazon.randomcutforest.CommonUtils.defaultScoreSeenFunction;
import static com.amazon.randomcutforest.CommonUtils.defaultScoreUnseenFunction;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.IBoundingBoxView;
import com.amazon.randomcutforest.tree.INodeView;
import com.amazon.randomcutforest.tree.NodeView;

public class ImputeVisitorTest {

    private float[] queryPoint;
    private int numberOfMissingValues;
    private int[] missingIndexes;
    private ImputeVisitor visitor;

    @BeforeEach
    public void setUp() {
        // create a point where the 2nd value is missing
        // The second value of queryPoint and the 2nd and 3rd values of missingIndexes
        // should be ignored in all tests

        queryPoint = new float[] { -1.0f, 1000.0f, 3.0f };
        numberOfMissingValues = 1;
        missingIndexes = new int[] { 1, 99, -888 };

        visitor = new ImputeVisitor(queryPoint, numberOfMissingValues, missingIndexes);
    }

    @Test
    public void testNew() {
        assertArrayEquals(toDoubleArray(queryPoint), visitor.getResult());
        assertNotSame(queryPoint, visitor.getResult());
        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, visitor.getAnomalyRank());
    }

    @Test
    public void testCopyConstructor() {
        ImputeVisitor copy = new ImputeVisitor(visitor);
        assertArrayEquals(toDoubleArray(queryPoint), copy.getResult());
        assertNotSame(copy.getResult(), visitor.getResult());
        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, visitor.getAnomalyRank());
    }

    @Test
    public void testAcceptLeafEquals() {
        float[] point = { queryPoint[0], 2.0f, queryPoint[2] };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getLiftedLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        visitor.acceptLeaf(leafNode, leafDepth);

        double[] expected = new double[] { -1.0, 2.0, 3.0 };
        assertArrayEquals(expected, visitor.getResult());

        assertEquals(defaultScoreSeenFunction(leafDepth, leafMass), visitor.getAnomalyRank());
    }

    @Test
    public void testAcceptLeafEqualsZeroDepth() {
        float[] point = { queryPoint[0], 2.0f, queryPoint[2] };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getLiftedLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafDepth = 0;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        visitor.acceptLeaf(leafNode, leafDepth);

        double[] expected = new double[] { -1.0, 2.0, 3.0 };
        assertArrayEquals(expected, visitor.getResult());

        assertEquals(0.0, visitor.getAnomalyRank());
    }

    @Test
    public void testAcceptLeafNotEquals() {
        float[] point = { queryPoint[0], 2.0f, -111.11f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getLiftedLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));
        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        visitor.acceptLeaf(leafNode, leafDepth);

        float[] expected = new float[] { -1.0f, 2.0f, 3.0f };
        assertArrayEquals(toDoubleArray(expected), visitor.getResult());

        assertEquals(defaultScoreUnseenFunction(leafDepth, leafMass), visitor.getAnomalyRank());
    }

    @Test
    public void testAccept() {

        float[] point = { queryPoint[0], 2.0f, -111.11f };
        INodeView node = mock(NodeView.class);
        when(node.getLeafPoint()).thenReturn(point);
        when(node.getLiftedLeafPoint()).thenReturn(point);
        when(node.getBoundingBox()).thenReturn(new BoundingBox(point, point));
        int depth = 100;
        int leafMass = 10;
        when(node.getMass()).thenReturn(leafMass);

        visitor.acceptLeaf(node, depth);

        double[] expected = new double[] { -1.0, 2.0, 3.0 };
        assertArrayEquals(expected, visitor.getResult());

        assertEquals(defaultScoreUnseenFunction(depth, leafMass), visitor.getAnomalyRank());

        depth--;
        IBoundingBoxView boundingBox = node.getBoundingBox().getMergedBox(new float[] { 99.0f, 4.0f, -19.0f });
        when(node.getBoundingBox()).thenReturn(boundingBox);
        when(node.getMass()).thenReturn(leafMass + 2);

        double oldRank = visitor.getAnomalyRank();
        visitor.accept(node, depth);
        assertArrayEquals(expected, visitor.getResult());

        double p = CommonUtils.getProbabilityOfSeparation(boundingBox, toFloatArray(expected));
        double expectedRank = p * defaultScoreUnseenFunction(depth, node.getMass()) + (1 - p) * oldRank;
        assertEquals(expectedRank, visitor.getAnomalyRank(), EPSILON);
    }

    @Test
    public void testNewCopy() {
        ImputeVisitor copy = (ImputeVisitor) visitor.newCopy();
        assertArrayEquals(toDoubleArray(queryPoint), copy.getResult());
        assertNotSame(copy.getResult(), visitor.getResult());
        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, visitor.getAnomalyRank());
    }

    @Test
    public void testMerge() {
        float[] otherPoint = new float[] { 99, 100, 101 };
        ImputeVisitor other = new ImputeVisitor(otherPoint, 0, new int[0]);

        // set other.rank to a small value
        NodeView node = mock(NodeView.class);
        when(node.getLeafPoint()).thenReturn(new float[] { 0, 0, 0 });
        when(node.getLiftedLeafPoint()).thenReturn(new float[] { 0, 0, 0 });
        when(node.getBoundingBox()).thenReturn(new BoundingBox(new float[] { 0, 0, 0 }));
        other.acceptLeaf(node, 99);

        assertTrue(other.getAnomalyRank() < visitor.getAnomalyRank());

        other.combine(visitor);
        assertArrayEquals(toDoubleArray(otherPoint), other.getResult());

        visitor.combine(other);
        assertArrayEquals(toDoubleArray(otherPoint), visitor.getResult());
    }
}
