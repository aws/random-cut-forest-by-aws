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

import static com.amazon.randomcutforest.CommonUtils.defaultScalarNormalizerFunction;
import static com.amazon.randomcutforest.CommonUtils.defaultScoreUnseenFunction;
import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.INodeView;
import com.amazon.randomcutforest.tree.NodeView;

public class AnomalyAttributionVisitorTest {

    @Test
    public void testNew() {
        double[] point = new double[] { 1.1, -2.2, 3.3 };
        int treeMass = 99;
        AnomalyAttributionVisitor visitor = new AnomalyAttributionVisitor(point, treeMass);

        assertFalse(visitor.pointInsideBox);
        for (int i = 0; i < point.length; i++) {
            assertFalse(visitor.coordInsideBox[i]);
        }

        assertFalse(visitor.ignoreLeaf);
        assertEquals(0, visitor.ignoreLeafMassThreshold);
        DiVector result = visitor.getResult();
        double[] zero = new double[point.length];
        assertArrayEquals(zero, result.high);
        assertArrayEquals(zero, result.low);
    }

    @Test
    public void testNewWithIgnoreOptions() {
        double[] point = new double[] { 1.1, -2.2, 3.3 };
        int treeMass = 99;
        AnomalyAttributionVisitor visitor = new AnomalyAttributionVisitor(point, treeMass, 7);

        assertFalse(visitor.pointInsideBox);
        for (int i = 0; i < point.length; i++) {
            assertFalse(visitor.coordInsideBox[i]);
        }

        assertTrue(visitor.ignoreLeaf);
        assertEquals(7, visitor.ignoreLeafMassThreshold);
        DiVector result = visitor.getResult();
        double[] zero = new double[point.length];
        assertArrayEquals(zero, result.high);
        assertArrayEquals(zero, result.low);
    }

    @Test
    public void testAcceptLeafEquals() {
        double[] point = { 1.1, -2.2, 3.3 };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int treeMass = 21;
        AnomalyAttributionVisitor visitor = new AnomalyAttributionVisitor(point, treeMass, 0);
        visitor.acceptLeaf(leafNode, leafDepth);

        assertTrue(visitor.hitDuplicates);
        double expectedScoreSum = CommonUtils.defaultDampFunction(leafMass, treeMass)
                / (leafDepth + Math.log(leafMass + 1) / Math.log(2));
        double expectedScore = expectedScoreSum / (2 * point.length);
        DiVector result = visitor.getResult();
        for (int i = 0; i < point.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.high[i], EPSILON);
        }
    }

    @Test
    public void testAcceptLeafNotEquals() {
        double[] point = new double[] { 1.1, -2.2, 3.3 };
        double[] anotherPoint = new double[] { -4.0, 5.0, 6.0 };

        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(anotherPoint);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(anotherPoint, anotherPoint));
        int leafDepth = 100;
        int leafMass = 4;
        when(leafNode.getMass()).thenReturn(leafMass);

        int treeMass = 21;
        AnomalyAttributionVisitor visitor = new AnomalyAttributionVisitor(point, treeMass, 0);
        visitor.acceptLeaf(leafNode, leafDepth);

        double expectedScoreSum = defaultScoreUnseenFunction(leafDepth, leafMass);
        double sumOfNewRange = (1.1 - (-4.0)) + (5.0 - (-2.2)) + (6.0 - 3.3);

        DiVector result = visitor.getResult();
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (1.1 - (-4.0)) / sumOfNewRange, treeMass),
                result.high[0], EPSILON);
        assertEquals(0.0, result.low[0]);
        assertEquals(0.0, result.high[1]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (5.0 - (-2.2)) / sumOfNewRange, treeMass),
                result.low[1], EPSILON);
        assertEquals(0.0, result.high[2]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (6.0 - 3.3) / sumOfNewRange, treeMass),
                result.low[2], EPSILON);

        visitor = new AnomalyAttributionVisitor(point, treeMass, 3);
        visitor.acceptLeaf(leafNode, leafDepth);
        result = visitor.getResult();
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (1.1 - (-4.0)) / sumOfNewRange, treeMass),
                result.high[0], EPSILON);
        assertEquals(0.0, result.low[0]);
        assertEquals(0.0, result.high[1]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (5.0 - (-2.2)) / sumOfNewRange, treeMass),
                result.low[1], EPSILON);
        assertEquals(0.0, result.high[2]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (6.0 - 3.3) / sumOfNewRange, treeMass),
                result.low[2], EPSILON);

        visitor = new AnomalyAttributionVisitor(point, treeMass, 4);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = expectedScoreSum / (2 * point.length);
        result = visitor.getResult();
        for (int i = 0; i < point.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.high[i], EPSILON);
        }
    }

    @Test
    public void testAccept() {
        double[] pointToScore = { 0.0, 0.0 };
        int treeMass = 50;
        AnomalyAttributionVisitor visitor = new AnomalyAttributionVisitor(pointToScore, treeMass, 0);

        INodeView leafNode = mock(NodeView.class);
        double[] point = new double[] { 1.0, -2.0 };
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafMass = 3;
        when(leafNode.getMass()).thenReturn(leafMass);
        int depth = 4;
        visitor.acceptLeaf(leafNode, depth);
        DiVector result = visitor.getResult();

        double expectedScoreSum = defaultScoreUnseenFunction(depth, leafNode.getMass());
        double sumOfNewRange = 1.0 + 2.0;

        double[] expectedUnnormalizedLow = new double[] { expectedScoreSum * 1.0 / sumOfNewRange, 0.0 };
        double[] expectedUnnormalizedHigh = new double[] { 0.0, expectedScoreSum * 2.0 / sumOfNewRange };

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow[i], treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh[i], treeMass), result.high[i],
                    EPSILON);
        }

        // parent does not contain pointToScore

        depth--;
        INodeView sibling = mock(NodeView.class);
        int siblingMass = 2;
        when(sibling.getMass()).thenReturn(siblingMass);
        INodeView parent = mock(NodeView.class);
        int parentMass = leafMass + siblingMass;
        when(parent.getMass()).thenReturn(parentMass);
        BoundingBox boundingBox = new BoundingBox(point, new double[] { 2.0, -0.5 });
        when(parent.getBoundingBox()).thenReturn(boundingBox);
        visitor.accept(parent, depth);
        result = visitor.getResult();

        double expectedSumOfNewRange2 = 2.0 + 2.0;
        double expectedProbOfCut2 = (1.0 + 0.5) / expectedSumOfNewRange2;
        double[] expectedDifferenceInRangeVector2 = { 0.0, 1.0, 0.5, 0.0 };

        double expectedScore2 = defaultScoreUnseenFunction(depth, parent.getMass());
        double[] expectedUnnormalizedLow2 = new double[pointToScore.length];
        double[] expectedUnnormalizedHigh2 = new double[pointToScore.length];

        for (int i = 0; i < pointToScore.length; i++) {
            double prob = expectedDifferenceInRangeVector2[2 * i] / expectedSumOfNewRange2;
            expectedUnnormalizedHigh2[i] = prob * expectedScore2
                    + (1 - expectedProbOfCut2) * expectedUnnormalizedHigh[i];

            prob = expectedDifferenceInRangeVector2[2 * i + 1] / expectedSumOfNewRange2;
            expectedUnnormalizedLow2[i] = prob * expectedScore2 + (1 - expectedProbOfCut2) * expectedUnnormalizedLow[i];
        }

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow2[i], treeMass), result.low[i],
                    EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh2[i], treeMass), result.high[i],
                    EPSILON);
        }

        // grandparent contains pointToScore

        assertFalse(visitor.pointInsideBox);

        depth--;
        INodeView grandParent = mock(NodeView.class);

        when(grandParent.getMass()).thenReturn(parentMass + 2);
        when(grandParent.getBoundingBox()).thenReturn(boundingBox
                .getMergedBox(new BoundingBox(new double[] { -1.0, 1.0 }).getMergedBox(new double[] { -0.5, -1.5 })));
        visitor.accept(grandParent, depth);
        result = visitor.getResult();

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow2[i], treeMass), result.low[i],
                    EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh2[i], treeMass), result.high[i],
                    EPSILON);
        }
    }

}
