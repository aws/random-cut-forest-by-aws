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

package com.amazon.randomcutforest.anomalydetection;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.IBoundingBoxView;
import com.amazon.randomcutforest.tree.INodeView;
import com.amazon.randomcutforest.tree.NodeView;

public class AnomalyScoreVisitorTest {

    @Test
    public void testNew() {
        float[] point = new float[] { 1.0f, 2.0f };
        int sampleSize = 9;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, sampleSize);

        assertFalse(visitor.pointInsideBox);
        for (int i = 0; i < point.length; i++) {
            assertFalse(visitor.coordInsideBox[i]);
        }

        assertFalse(visitor.ignoreLeafEquals);
        assertEquals(0, visitor.ignoreLeafMassThreshold);
        assertThat(visitor.getResult(), is(0.0));
    }

    @Test
    public void testNewWithIgnoreOptions() {
        float[] point = new float[] { 1.0f, 2.0f };
        int sampleSize = 9;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, sampleSize, 7);

        assertFalse(visitor.pointInsideBox);
        for (int i = 0; i < point.length; i++) {
            assertFalse(visitor.coordInsideBox[i]);
        }

        assertTrue(visitor.ignoreLeafEquals);
        assertEquals(7, visitor.ignoreLeafMassThreshold);
        assertThat(visitor.getResult(), is(0.0));
    }

    @Test
    public void testAcceptLeafEquals() {
        float[] point = { 1.0f, 2.0f, 3.0f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int subSampleSize = 21;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, subSampleSize);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = CommonUtils.defaultDampFunction(leafMass, subSampleSize)
                / (leafDepth + Math.log(leafMass + 1) / Math.log(2));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, subSampleSize), EPSILON));
        assertTrue(visitor.pointInsideBox);

        visitor = new AnomalyScoreVisitor(point, subSampleSize);
        visitor.acceptLeaf(leafNode, 0);
        expectedScore = CommonUtils.defaultDampFunction(leafMass, subSampleSize)
                / (Math.log(leafMass + 1) / Math.log(2.0));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, subSampleSize), EPSILON));
        assertTrue(visitor.pointInsideBox);
    }

    @Test
    public void testAcceptLeafNotEquals() {
        float[] point = new float[] { 1.0f, 2.0f, 3.0f };
        float[] anotherPoint = new float[] { 4.0f, 5.0f, 6.0f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(anotherPoint);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(anotherPoint, anotherPoint));

        int leafDepth = 100;

        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, 2);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = 1.0 / (leafDepth + 1);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, 2), EPSILON));
        assertFalse(visitor.pointInsideBox);
    }

    @Test
    public void testAcceptEqualsLeafPoint() {
        float[] pointToScore = { 0.0f, 0.0f };
        int sampleSize = 50;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(pointToScore, sampleSize);

        float[] point = Arrays.copyOf(pointToScore, pointToScore.length);
        INodeView node = mock(NodeView.class);
        when(node.getLeafPoint()).thenReturn(point);
        when(node.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int depth = 2;
        visitor.acceptLeaf(node, depth);
        double expectedScore = CommonUtils.defaultDampFunction(node.getMass(), sampleSize)
                / (depth + Math.log(node.getMass() + 1) / Math.log(2));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        IBoundingBoxView boundingBox = node.getBoundingBox().getMergedBox(new float[] { 1.0f, 1.0f });
        node = new NodeView(null, null, Null);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new float[] { -1.0f, -1.0f });
        node = new NodeView(null, null, Null);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
    }

    @Test
    public void testAccept() {
        float[] pointToScore = new float[] { 0.0f, 0.0f };
        int sampleSize = 50;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(pointToScore, sampleSize);

        NodeView node = mock(NodeView.class);
        float[] otherPoint = new float[] { 1.0f, 1.0f };
        when(node.getLeafPoint()).thenReturn(otherPoint);
        when(node.getBoundingBox()).thenReturn(new BoundingBox(otherPoint, otherPoint));
        int depth = 4;
        visitor.acceptLeaf(node, depth);
        double expectedScore = 1.0 / (depth + 1);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        IBoundingBoxView boundingBox = node.getBoundingBox().getMergedBox(new float[] { 2.0f, 0.0f });
        when(node.getBoundingBox()).thenReturn(boundingBox);
        when(node.probailityOfSeparation(any())).thenReturn(1.0 / 3);
        visitor.accept(node, depth);
        double p = visitor.getProbabilityOfSeparation(boundingBox);
        expectedScore = p * (1.0 / (depth + 1)) + (1 - p) * expectedScore;
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new float[] { -1.0f, 0.0f });

        when(node.getBoundingBox()).thenReturn(boundingBox);
        when(node.probailityOfSeparation(any())).thenReturn(0.0);
        visitor.accept(node, depth);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        expectedScore = p * (1.0 / (depth + 1)) + (1 - p) * expectedScore;
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new float[] { -1.0f, -1.0f });
        when(node.probailityOfSeparation(any())).thenReturn(0.0);
        visitor.accept(node, depth);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
        assertTrue(visitor.pointInsideBox);
    }

    @Test
    public void testGetProbabilityOfSeparation() {
        float[] minPoint = { 0.0f, 0.0f, 0.0f };
        float[] maxPoint = { 1.0f, 2.0f, 3.0f };
        IBoundingBoxView boundingBox = new BoundingBox(minPoint);
        boundingBox = boundingBox.getMergedBox(maxPoint);

        float[] point = { 0.5f, 0.5f, 0.5f };
        int sampleSize = 2;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, sampleSize);

        double p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo(0.0, EPSILON));
        assertTrue(visitor.coordInsideBox[0]);
        assertTrue(visitor.coordInsideBox[1]);
        assertTrue(visitor.coordInsideBox[2]);

        visitor = new AnomalyScoreVisitor(point, sampleSize);
        visitor.coordInsideBox[1] = visitor.coordInsideBox[2] = true;
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo(0.0, EPSILON));
        assertTrue(visitor.coordInsideBox[0]);
        assertTrue(visitor.coordInsideBox[1]);
        assertTrue(visitor.coordInsideBox[2]);

        point = new float[] { 2.0f, 0.5f, 0.5f };
        visitor = new AnomalyScoreVisitor(point, sampleSize);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo(1.0 / (2.0 + 2.0 + 3.0), EPSILON));
        assertFalse(visitor.coordInsideBox[0]);
        assertTrue(visitor.coordInsideBox[1]);
        assertTrue(visitor.coordInsideBox[2]);

        visitor = new AnomalyScoreVisitor(point, sampleSize);
        visitor.coordInsideBox[1] = visitor.coordInsideBox[2] = true;
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo(1.0 / (2.0 + 2.0 + 3.0), EPSILON));
        assertFalse(visitor.coordInsideBox[0]);
        assertTrue(visitor.coordInsideBox[1]);
        assertTrue(visitor.coordInsideBox[2]);

        point = new float[] { 0.5f, -3.0f, 4.0f };
        visitor = new AnomalyScoreVisitor(point, sampleSize);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo((3.0 + 1.0) / (1.0 + 5.0 + 4.0), EPSILON));
        assertTrue(visitor.coordInsideBox[0]);
        assertFalse(visitor.coordInsideBox[1]);
        assertFalse(visitor.coordInsideBox[2]);

        visitor = new AnomalyScoreVisitor(point, sampleSize);
        visitor.coordInsideBox[0] = true;
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(p, closeTo((3.0 + 1.0) / (1.0 + 5.0 + 4.0), EPSILON));
        assertTrue(visitor.coordInsideBox[0]);
        assertFalse(visitor.coordInsideBox[1]);
        assertFalse(visitor.coordInsideBox[2]);
    }

    @Test
    public void test_getProbabilityOfSeparation_leafNode() {
        float[] point = new float[] { 1.0f, 2.0f, 3.0f };
        float[] leafPoint = Arrays.copyOf(point, point.length);
        BoundingBox boundingBox = new BoundingBox(leafPoint);

        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, 2);
        assertThrows(IllegalStateException.class, () -> visitor.getProbabilityOfSeparation(boundingBox));
    }
}
