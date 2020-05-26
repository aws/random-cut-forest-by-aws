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
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Node;

public class AnomalyScoreVisitorTest {

    @Test
    public void testNew() {
        double[] point = new double[] { 1.0, 2.0 };
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
        double[] point = new double[] { 1.0, 2.0 };
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
        double[] point = { 1.0, 2.0, 3.0 };
        Node leafNode = spy(new Node(point));

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
        double[] point = new double[] { 1.0, 2.0, 3.0 };
        double[] anotherPoint = new double[] { 4.0, 5.0, 6.0 };

        Node leafNode = new Node(anotherPoint);
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
        double[] pointToScore = { 0.0, 0.0 };
        int sampleSize = 50;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(pointToScore, sampleSize);

        double[] point = Arrays.copyOf(pointToScore, pointToScore.length);
        Node node = new Node(point);
        int depth = 2;
        visitor.acceptLeaf(node, depth);
        double expectedScore = CommonUtils.defaultDampFunction(node.getMass(), sampleSize)
                / (depth + Math.log(node.getMass() + 1) / Math.log(2));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        BoundingBox boundingBox = node.getBoundingBox().getMergedBox(new double[] { 1.0, 1.0 });
        node = new Node(null, null, null, boundingBox);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new double[] { -1.0, -1.0 });
        node = new Node(null, null, null, boundingBox);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
    }

    @Test
    public void testAccept() {
        double[] pointToScore = new double[] { 0.0, 0.0 };
        int sampleSize = 50;
        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(pointToScore, sampleSize);

        Node node = new Node(new double[] { 1.0, 1.0 });
        int depth = 4;
        visitor.acceptLeaf(node, depth);
        double expectedScore = 1.0 / (depth + 1);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        BoundingBox boundingBox = node.getBoundingBox().getMergedBox(new double[] { 2.0, 0.0 });
        node = new Node(null, null, null, boundingBox);
        visitor.accept(node, depth);
        double p = visitor.getProbabilityOfSeparation(boundingBox);
        expectedScore = p * (1.0 / (depth + 1)) + (1 - p) * expectedScore;
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new double[] { -1.0, 0.0 });
        node = new Node(null, null, null, boundingBox);
        visitor.accept(node, depth);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        expectedScore = p * (1.0 / (depth + 1)) + (1 - p) * expectedScore;
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new double[] { -1.0, -1.0 });
        node = new Node(null, null, null, boundingBox);
        visitor.accept(node, depth);
        p = visitor.getProbabilityOfSeparation(boundingBox);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
        assertTrue(visitor.pointInsideBox);
    }

    @Test
    public void testGetProbabilityOfSeparation() {
        double[] minPoint = { 0.0, 0.0, 0.0 };
        double[] maxPoint = { 1.0, 2.0, 3.0 };
        BoundingBox boundingBox = new BoundingBox(minPoint);
        boundingBox = boundingBox.getMergedBox(maxPoint);

        double[] point = { 0.5, 0.5, 0.5 };
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

        point = new double[] { 2.0, 0.5, 0.5 };
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

        point = new double[] { 0.5, -3.0, 4.0 };
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
        double[] point = new double[] { 1.0, 2.0, 3.0 };
        double[] leafPoint = Arrays.copyOf(point, point.length);
        BoundingBox boundingBox = new BoundingBox(leafPoint);

        AnomalyScoreVisitor visitor = new AnomalyScoreVisitor(point, 2);
        assertThrows(IllegalStateException.class, () -> visitor.getProbabilityOfSeparation(boundingBox));
    }
}
