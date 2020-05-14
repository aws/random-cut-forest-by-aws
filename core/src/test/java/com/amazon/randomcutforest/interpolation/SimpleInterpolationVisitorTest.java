/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.interpolation;

import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Node;
import com.amazon.randomcutforest.tree.Cut;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

public class SimpleInterpolationVisitorTest {

    private static final int SEED = 1002;

    @Test
    public void testNew() {
        double[] point = {1.0, 2.0};
        int sampleSize = 9;
        SimpleInterpolationVisitor visitor = new SimpleInterpolationVisitor(point, sampleSize,1,false);

        assertFalse(visitor.pointInsideBox);
        assertEquals(2, visitor.coordInsideBox.length);

        for (int i = 0; i < point.length; i++) {
            assertFalse(visitor.coordInsideBox[i]);
        }

        InterpolationMeasure output = visitor.getResult();

        double[] zero = new double[point.length];

        assertArrayEquals(zero, output.measure.high);
        assertArrayEquals(zero, output.distances.high);
        assertArrayEquals(zero, output.probMass.high);
        assertArrayEquals(zero, output.measure.low);
        assertArrayEquals(zero, output.distances.low);
        assertArrayEquals(zero, output.probMass.low);
    }

    @Test
    public void testAcceptLeafEquals() {
        double[] point = {1.0, 2.0, 3.0};
        Node leafNode = Mockito.spy(new Node(point));

        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int sampleSize = 21;
        SimpleInterpolationVisitor visitor = new SimpleInterpolationVisitor(point, sampleSize,1,false);
        visitor.acceptLeaf(leafNode, leafDepth);

        InterpolationMeasure result = visitor.getResult();

        double[] expected = new double[point.length];
        Arrays.fill(expected, 0.5 * (1 + leafMass) / point.length);
        assertArrayEquals(expected, result.measure.high);
        assertArrayEquals(expected, result.measure.low);

        Arrays.fill(expected, 0.5 / point.length);
        assertArrayEquals(expected, result.probMass.high);
        assertArrayEquals(expected, result.probMass.low);

        Arrays.fill(expected, 0.0);
        assertArrayEquals(expected, result.distances.high);
        assertArrayEquals(expected, result.distances.low);
    }

    @Test
    public void testAcceptLeafNotEquals() {
        double[] point = {1.0, 9.0, 4.0};
        double[] anotherPoint = {4.0, 5.0, 6.0};

        Node leafNode = Mockito.spy(new Node(anotherPoint));
        when(leafNode.getMass()).thenReturn(4);
        int leafDepth = 100;
        int sampleSize = 99;

        SimpleInterpolationVisitor visitor = new SimpleInterpolationVisitor(point, sampleSize,1,false);
        visitor.acceptLeaf(leafNode, leafDepth);

        InterpolationMeasure result = visitor.getResult();

        double expectedSumOfNewRange = 3.0 + 4.0 + 2.0;
        double[] expectedDifferenceInRangeVector = {0.0, 3.0, 4.0, 0.0, 0.0, 2.0};
        double[] expectedProbVector = Arrays.stream(expectedDifferenceInRangeVector)
                .map(x -> x / expectedSumOfNewRange)
                .toArray();
        double[] expectedmeasure = Arrays.stream(expectedProbVector).toArray();

        double[] expectedDistances = new double[2 * point.length];
        for (int i = 0; i < 2 * point.length; i++) {
            expectedDistances[i] = expectedProbVector[i] * expectedDifferenceInRangeVector[i];
        }
        for (int i = 0; i < 2 * point.length; i++) {
            expectedmeasure[i]=expectedmeasure[i]*5;
        }
        for(int i=0;i<point.length;i++) {
            assertEquals(expectedProbVector[2*i], result.probMass.high[i]);
            assertEquals(expectedProbVector[2*i+1], result.probMass.low[i]);

            assertEquals(expectedmeasure[2*i], result.measure.high[i]);
            assertEquals(expectedmeasure[2*i+1], result.measure.low[i]);

            assertEquals(expectedDistances[2*i], result.distances.high[i]);
            assertEquals(expectedDistances[2*i+1], result.distances.low[i]);
        }

    }

    @Test
    public void testAcceptEqualsLeafPoint() {
        double[] pointToScore = {0.0, 0.0};
        int sampleSize = 50;
        SimpleInterpolationVisitor visitor = new SimpleInterpolationVisitor(pointToScore, sampleSize,1,false);

        double[] point = Arrays.copyOf(pointToScore, pointToScore.length);
        Node node = new Node(point);
        int depth = 2;
        visitor.acceptLeaf(node, depth);
        InterpolationMeasure result = visitor.getResult();

        double[] expected = new double[point.length];
        Arrays.fill(expected, 0.5 * (1 + node.getMass()) / point.length);
        assertArrayEquals(expected, result.measure.high);
        assertArrayEquals(expected, result.measure.low);

        Arrays.fill(expected, 0.5 / point.length);
        assertArrayEquals(expected, result.probMass.high);
        assertArrayEquals(expected, result.probMass.low);

        Arrays.fill(expected, 0.0);
        assertArrayEquals(expected, result.distances.high);
        assertArrayEquals(expected, result.distances.low);

        depth--;
        double[] siblingPoint = {1.0, -2.0};
        Node sibling = Mockito.spy(new Node(siblingPoint));
        int siblingMass = 2;
        when(sibling.getMass()).thenReturn(siblingMass);
        Node parent = Mockito.spy(new Node(node, sibling, new Cut(0, 0.5),
                node.getBoundingBox().getMergedBox(sibling.getBoundingBox())));
        when(parent.getMass()).thenReturn(node.getMass() + siblingMass);
        visitor.accept(parent, depth);
        result = visitor.getResult();

        // compute using shadow box (sibling leaf node at {1.0, -2.0} and parent bounding box

        double[] directionalDistance = {0.0, 1.0, 2.0, 0.0};
        double[] differenceInRange = {0.0, 1.0, 2.0, 0.0};
        double sumOfNewRange = 1.0 + 2.0;
        double[] probVector = Arrays.stream(differenceInRange)
                .map(x -> x / sumOfNewRange)
                .toArray();
        expected=new double [2*pointToScore.length];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = probVector[i] * (1 + node.getMass() + parent.getMass());
        }
        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(expected[2*i], result.measure.high[i]);
            assertEquals(expected[2*i+1], result.measure.low[i]);
        }


        for (int i = 0; i < expected.length; i++) {
            expected[i] = probVector[i];
        }

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(expected[2*i], result.probMass.high[i]);
            assertEquals(expected[2*i+1], result.probMass.low[i]);
        }

        for (int i = 0; i < expected.length; i++) {
            expected[i] = probVector[i] * directionalDistance[i];
        }
        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(expected[2*i], result.distances.high[i]);
            assertEquals(expected[2*i+1], result.distances.low[i]);
        }

    }

    @Test
    public void testAccept() {
        double[] pointToScore = {0.0, 0.0};
        int sampleSize = 50;
        SimpleInterpolationVisitor visitor = new SimpleInterpolationVisitor(pointToScore, sampleSize,1,false);

        Node leafNode = Mockito.spy(new Node(new double[] {1.0, -2.0}));
        int leafMass = 3;
        when(leafNode.getMass()).thenReturn(leafMass);
        int depth = 4;
        visitor.acceptLeaf(leafNode, depth);
        InterpolationMeasure result = visitor.getResult();

        double expectedSumOfNewRange = 1.0 + 2.0;
        double[] expectedDifferenceInRangeVector = {0.0, 1.0, 2.0, 0.0};
        double[] expectedProbVector = Arrays.stream(expectedDifferenceInRangeVector)
                .map(x -> x / expectedSumOfNewRange)
                .toArray();
        double[] expectedNumPts = Arrays.stream(expectedProbVector)
                .toArray();

        double[] expectedDistances = new double[2 * pointToScore.length];
        for (int i = 0; i < 2 * pointToScore.length; i++) {
            expectedDistances[i] = expectedProbVector[i] * expectedDifferenceInRangeVector[i];
        }

        for (int i = 0; i < 2 * pointToScore.length; i++) {
            expectedNumPts[i] = expectedNumPts[i] * 4;
        }

        for(int i=0;i<pointToScore.length;i++) {
            assertEquals(expectedProbVector[2*i], result.probMass.high[i]);
            assertEquals(expectedProbVector[2*i+1], result.probMass.low[i]);

            assertEquals(expectedNumPts[2*i], result.measure.high[i]);
            assertEquals(expectedNumPts[2*i+1], result.measure.low[i]);

            assertEquals(expectedDistances[2*i], result.distances.high[i]);
            assertEquals(expectedDistances[2*i+1], result.distances.low[i]);
        }

        // parent does not contain pointToScore

        depth--;
        Node sibling = Mockito.spy(new Node(new double[] {2.0, -0.5}));
        int siblingMass = 2;
        when(sibling.getMass()).thenReturn(siblingMass);
        Node parent = Mockito.spy(new Node(leafNode, sibling, new Cut(0, 0.5),
                leafNode.getBoundingBox().getMergedBox(sibling.getBoundingBox())));
        int parentMass = leafMass + siblingMass;
        when(parent.getMass()).thenReturn(parentMass);
        visitor.accept(parent, depth);
        result = visitor.getResult();

        double expectedSumOfNewRange2 = 2.0 + 2.0;
        double expectedProbOfCut2 = (1.0 + 0.5) / expectedSumOfNewRange2;
        double[] expectedDifferenceInRangeVector2 = {0.0, 1.0, 0.5, 0.0};
        double[] expectedDirectionalDistanceVector2 = {0.0, 2.0, 2.0, 0.0};

        for (int i = 0; i < 2 * pointToScore.length; i++) {
            double prob = expectedDifferenceInRangeVector2[i] / expectedSumOfNewRange2;
            expectedProbVector[i] = prob + (1 - expectedProbOfCut2) * expectedProbVector[i];
            expectedNumPts[i] = prob * (1 + parent.getMass()) + (1 - expectedProbOfCut2) * expectedNumPts[i];
            expectedDistances[i] = prob * expectedDirectionalDistanceVector2[i]
                    + (1 - expectedProbOfCut2) * expectedDistances[i];
        }

        for(int i=0;i<pointToScore.length;i++) {
            assertEquals(expectedProbVector[2*i], result.probMass.high[i]);
            assertEquals(expectedProbVector[2*i+1], result.probMass.low[i]);

            assertEquals(expectedNumPts[2*i], result.measure.high[i]);
            assertEquals(expectedNumPts[2*i+1], result.measure.low[i]);

            assertEquals(expectedDistances[2*i], result.distances.high[i]);
            assertEquals(expectedDistances[2*i+1], result.distances.low[i]);
        }

        // grandparent contains pointToScore

        assertFalse(visitor.pointInsideBox);

        depth--;
        Node aunt = new Node(null, null, new Cut(1, 0.5),
                new BoundingBox(new double[] {-1.0, 1.0}).getMergedBox(new double[] {-0.5, -1.5}));
        Node grandparent = Mockito.spy(new Node(parent, aunt, new Cut(0, 0.1),
                parent.getBoundingBox().getMergedBox(aunt.getBoundingBox())));
        when(grandparent.getMass()).thenReturn(parentMass + aunt.getMass());
        visitor.accept(grandparent, depth);
        result = visitor.getResult();

        for(int i=0;i<pointToScore.length;i++) {
            assertEquals(expectedProbVector[2*i], result.probMass.high[i]);
            assertEquals(expectedProbVector[2*i+1], result.probMass.low[i]);

            assertEquals(expectedNumPts[2*i], result.measure.high[i]);
            assertEquals(expectedNumPts[2*i+1], result.measure.low[i]);

            assertEquals(expectedDistances[2*i], result.distances.high[i]);
            assertEquals(expectedDistances[2*i+1], result.distances.low[i]);
        }

        assertTrue(visitor.pointInsideBox);
    }
}
