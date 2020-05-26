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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

public class NodeTest {

    @Test
    public void testNewParentNode() {
        Node leftChild = new Node(new double[] { 0.0, 0.0 });
        Node rightChild = new Node(new double[] { 10.0, 10.0 });
        Cut cut = new Cut(0, 5.0);
        BoundingBox box = new BoundingBox(new double[] { 0.0, 0.0 }).getMergedBox(new double[] { 10.0, 10.0 });
        Node node = new Node(leftChild, rightChild, cut, box, true);

        assertThat(node.getLeftChild(), is(leftChild));
        assertThat(node.getRightChild(), is(rightChild));
        assertThat(node.getCut(), is(cut));
        assertThat(node.getBoundingBox(), is(box));
        assertThat(node.isLeaf(), is(false));
        assertThat(node.getMass(), is(0));
        assertThat(node.getParent(), is(nullValue()));
        assertThrows(IllegalStateException.class, node::getLeafPoint);
        assertThrows(IllegalStateException.class, () -> node.getLeafPoint(0));
    }

    @Test
    public void testNewLeafNode() {
        double[] leafPoint = { -11.0, -12.0 };
        Node node = new Node(leafPoint);
        assertThat(node.getMass(), is(0));
        assertThat(node.getRightChild(), is(nullValue()));
        assertThat(node.getCut(), is(nullValue()));

        BoundingBox expectedBox = new BoundingBox(leafPoint);
        assertThat(node.getBoundingBox(), is(expectedBox));

        assertThat(node.isLeaf(), is(true));
        assertTrue(node.leafPointEquals(Arrays.copyOf(leafPoint, leafPoint.length)));
        assertThat(node.getMass(), is(0));
        assertThat(node.getParent(), is(nullValue()));

        assertArrayEquals(leafPoint, node.getLeafPoint());
        for (int i = 0; i < leafPoint.length; i++) {
            assertEquals(leafPoint[i], node.getLeafPoint(i));
        }

        Node child = new Node(new double[] { 13.0, 14.0 });
        assertThrows(IllegalStateException.class, () -> node.setLeftChild(child));
        assertThrows(IllegalStateException.class, () -> node.setRightChild(child));
    }

    @Test
    public void testMassOperations() {
        double[] leafPoint = { -11.0, -12.0 };
        Node node = new Node(leafPoint);
        assertThat(node.getMass(), is(0));

        node.setMass(-11);
        assertThat(node.getMass(), is(-11));

        node.addMass(100);
        assertThat(node.getMass(), is(89));

        node.incrementMass();
        assertThat(node.getMass(), is(90));

        node.decrementMass();
        assertThat(node.getMass(), is(89));
    }

    @Test
    public void testCenterOfMassOperations() {
        Node leftChild = new Node(new double[] { 0.0, 0.0 });
        Node rightChild = new Node(new double[] { 10.0, 10.0 });
        Cut cut = new Cut(0, 5.0);
        BoundingBox box = new BoundingBox(new double[] { 0.0, 0.0 }).getMergedBox(new double[] { 10.0, 10.0 });
        Node node = new Node(leftChild, rightChild, cut, box, true);
        node.addMass(2);
        node.addToPointSum(new double[] { 10.0, 10.0 });

        assertArrayEquals(new double[] { 10.0, 10.0 }, node.getPointSum(), EPSILON);
        assertArrayEquals(new double[] { 5.0, 5.0 }, node.getCenterOfMass(), EPSILON);

        node.addToPointSum(new double[] { 1.0, -1.0 });
        node.incrementMass();
        assertArrayEquals(new double[] { 11.0, 9.0 }, node.getPointSum(), EPSILON);
        assertArrayEquals(new double[] { 11.0 / 3, 9.0 / 3 }, node.getCenterOfMass(), EPSILON);

        node.subtractFromPointSum(new double[] { 10.0, 10.0 });
        node.decrementMass();
        assertArrayEquals(new double[] { 1.0, -1.0 }, node.getPointSum(), EPSILON);
        assertArrayEquals(new double[] { 0.5, -0.5 }, node.getCenterOfMass(), EPSILON);

        double[] leafPoint = { -11.0, -12.0 };
        Node leafNode = new Node(leafPoint);
        leafNode.setMass(2);
        assertArrayEquals(new double[] { -22.0, -24.0 }, leafNode.getPointSum());
        assertArrayEquals(leafPoint, leafNode.getCenterOfMass());
    }

    @Test
    public void testCenterOfMassDisabled() {
        Node leftChild = new Node(new double[] { 0.0, 0.0 });
        Node rightChild = new Node(new double[] { 10.0, 10.0 });
        Cut cut = new Cut(0, 5.0);
        BoundingBox box = new BoundingBox(new double[] { 0.0, 0.0 }).getMergedBox(new double[] { 10.0, 10.0 });
        Node node = new Node(leftChild, rightChild, cut, box, false);
        node.addMass(2);

        assertThrows(IllegalStateException.class, () -> node.addToPointSum(new double[] { 10.0, 10.0 }));
        assertThrows(IllegalStateException.class, () -> node.subtractFromPointSum(new double[] { 10.0, 10.0 }));

        double[] zero = new double[2];
        assertArrayEquals(zero, node.getPointSum());
        assertArrayEquals(zero, node.getCenterOfMass());
    }

    @Test
    public void testSequenceIndexOperations() {
        double[] leafPoint = { -11.0, -12.0 };
        Node node = new Node(leafPoint);

        node.addSequenceIndex(123L);
        node.addSequenceIndex(456L);
        node.addSequenceIndex(789L);
        assertThat(node.getSequenceIndexes(), containsInAnyOrder(123L, 456L, 789L));

        node.deleteSequenceIndex(456L);
        assertThat(node.getSequenceIndexes(), containsInAnyOrder(123L, 789L));
        assertArrayEquals(leafPoint, node.getCenterOfMass());

        Node nodeWithSequenceIndexesDisabled = new Node(leafPoint);
        assertTrue(nodeWithSequenceIndexesDisabled.getSequenceIndexes().isEmpty());
    }
}
