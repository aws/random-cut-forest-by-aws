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

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.store.PointStoreDouble;

public class CompactRandomCutTreeDoubleTest {

    private static final double EPSILON = 1e-8;

    private Random rng;
    private CompactRandomCutTreeDouble tree;
    private int capacity = 100;
    int defaultTreeSize = RandomCutForest.DEFAULT_SAMPLE_SIZE;

    @BeforeEach
    public void setUp() {
        rng = mock(Random.class);
        PointStoreDouble pointStoreDouble = new PointStoreDouble.Builder().indexCapacity(capacity).capacity(capacity)
                .currentStoreCapacity(capacity).dimensions(2).build();
        tree = CompactRandomCutTreeDouble.builder().random(rng).centerOfMassEnabled(true).pointStore(pointStoreDouble)
                .storeSequenceIndexesEnabled(true).build();

        assertEquals(tree.getNodeStore().getCapacity(), defaultTreeSize - 1);
        // Create the following tree structure (in the second diagram., backticks denote
        // cuts)
        // The leaf point 0,1 has mass 2, all other nodes have mass 1.
        //
        // /\
        // / \
        // -1,-1 / \
        // / \
        // /\ 1,1
        // / \
        // -1,0 0,1
        //
        //
        // 0,1 1,1
        // ----------*---------*
        // | ` | ` |
        // | ` | ` |
        // | ` | ` |
        // -1,0 *-------------------|
        // | |
        // |```````````````````|
        // | |
        // -1,-1 *--------------------
        //
        // We choose the insertion order and random draws carefully so that each split
        // divides its parent in half.
        // The random values are used to set the cut dimensions and values.

        assertArrayEquals(tree.liftFromTree(new double[] { 1, 2, 3, 4 }), new double[] { 1, 2, 3, 4 }, EPSILON);

        assertEquals(pointStoreDouble.add(new double[] { -1, -1 }, 1), 0);
        assertEquals(pointStoreDouble.add(new double[] { 1, 1 }, 2), 1);
        assertEquals(pointStoreDouble.add(new double[] { -1, 0 }, 3), 2);
        assertEquals(pointStoreDouble.add(new double[] { 0, 1 }, 4), 3);
        assertEquals(pointStoreDouble.add(new double[] { 0, 1 }, 5), 4);
        assertEquals(pointStoreDouble.add(new double[] { 0, 0 }, 6), 5);

        assertEquals(tree.getEquivalentReference(3), null);
        tree.addPoint(0, 1);
        assertEquals(tree.getRoot(), defaultTreeSize - 1);
        assertTrue(tree.isLeaf(tree.getRoot()));
        tree.reorderNodesInBreadthFirstOrder();
        assertEquals(tree.getRoot(), defaultTreeSize - 1);

        when(rng.nextDouble()).thenReturn(0.625);
        tree.addPoint(1, 2);

        when(rng.nextDouble()).thenReturn(0.5);
        tree.addPoint(2, 3);

        when(rng.nextDouble()).thenReturn(0.25);
        tree.addPoint(3, 4);

        // add mass to 0,1
        assertEquals(tree.addPoint(4, 5), 3);

        assertEquals(tree.findLeafAndVerify(4), tree.findLeafAndVerify(3));
        assertEquals(tree.getCopiesOfReference(3), 2);
        tree.setLeafPointReference(tree.findLeafAndVerify(4), 4);
        assertEquals(tree.getEquivalentReference(3), 4);
        tree.switchLeafReference(3);
        assertEquals(tree.getEquivalentReference(3), 3);
        assertEquals(tree.getEquivalentReference(5), null);
    }

    /**
     * Verify that the tree has the form described in the setUp method.
     */
    @Test
    public void testInitialTreeState() {
        int node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(5));
        assertArrayEquals(new double[] { -1, 2 }, tree.getPointSum(node), EPSILON);
        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(1L), 1);

        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(4));
        assertArrayEquals(new double[] { 0.0, 3.0 }, tree.getPointSum(node), EPSILON);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(2L), 1);

        node = tree.getLeftChild(node);
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));

        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));
        assertArrayEquals(new double[] { -1.0, 2.0 }, tree.getPointSum(node), EPSILON);
        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(2));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(4L), 1);
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(5L), 1);

        assertThrows(IllegalStateException.class, () -> tree.deletePoint(5, 6));
    }

    @Test
    public void testDeletePointWithLeafSibling() {
        tree.deletePoint(2, 3);

        // root node bounding box and cut remains unchanged, mass and centerOfMass are
        // updated

        int node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { 0.0, 2.0 }, tree.getPointSum(node), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(1L), 1);

        // sibling node moves up and bounding box recomputed

        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new double[] { 0, 1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));
        assertArrayEquals(new double[] { 1.0, 3.0 }, tree.getPointSum(node), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { 0, 1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(2));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(4L), 1);
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(5L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(2L), 1);

        tree.addPoint(4, 5);
        assertEquals(tree.getMass(tree.getLeftChild(node)), 3);
        tree.deletePoint(4, 5);
        assertEquals(tree.getMass(tree.getLeftChild(node)), 2);
    }

    @Test
    public void testDeletePointWithNonLeafSibling() {
        tree.deletePoint(1, 2);

        // root node bounding box recomputed

        int node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 0, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(1L), 1);

        // sibling node moves up and bounding box stays the same

        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(2));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(4L), 1);
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(5L), 1);
    }

    @Test
    public void testDeletePointWithMassGreaterThan1() {
        tree.deletePoint(3, 4);

        // same as initial state except mass at 0,1 is 1

        int node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(1L), 1);
        assertArrayEquals(new double[] { -1.0, 1.0 }, tree.getPointSum(), EPSILON);

        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 1, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));

        assertArrayEquals(new double[] { 0.0, 2.0 }, tree.getPointSum(node), EPSILON);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(2L), 1);

        node = tree.getLeftChild(node);
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(tree.getBoundingBox(node), is(expectedBox));
        assertEquals(expectedBox.toString(), tree.getBoundingBox(node).toString());
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { -1.0, 1.0 }, tree.getPointSum(node), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.getEquivalentReference(4), 3);
        assertEquals(tree.getCopiesOfReference(4), 0);
        assertEquals(tree.getCopiesOfReference(3), 1);
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(5L), 1);
    }

    @Test
    public void testRemap() {
        tree.deletePoint(0, 1);

        assertNotEquals(tree.getRoot(), 0);
        tree.reorderNodesInBreadthFirstOrder();
        assertEquals(tree.getRoot(), 0);
        int node = tree.getRoot();
        assertThat(tree.getMass(node), is(4));
        assertEquals(tree.getRightChild(node), defaultTreeSize - 1);
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(2L), 1);

        node = tree.getLeftChild(node);
        assertEquals(node, 1);
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));
        assertEquals(tree.getLeftChild(node), defaultTreeSize);
        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.getPoint(tree.getLeftChild(node)), is(new double[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getLeftChild(node))].get(3L), 1);

        assertEquals(tree.getRightChild(node), defaultTreeSize + 1);
        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.getPoint(tree.getRightChild(node)), is(new double[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(2));
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(5L), 1);
        assertEquals(tree.sequenceIndexes[tree.nodeStore.computeLeafIndex(tree.getRightChild(node))].get(4L), 1);
    }

    @Test
    public void testDeletePointInvalid() {
        // specified sequence index does not exist
        assertThrows(IllegalStateException.class, () -> tree.deletePoint(2, 99));

        // point does not exist in tree
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(7, 3));
    }

    @Test
    public void testUpdatesOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        PointStoreDouble pointStoreDouble = new PointStoreDouble.Builder().indexCapacity(capacity).capacity(capacity)
                .currentStoreCapacity(capacity).dimensions(1).build();
        CompactRandomCutTreeDouble tree = CompactRandomCutTreeDouble.builder().random(rng).pointStore(pointStoreDouble)
                .build();

        List<Weighted<double[]>> points = new ArrayList<>();
        points.add(new Weighted<>(new double[] { 48.08 }, 0, 1L));
        points.add(new Weighted<>(new double[] { 48.08000000000001 }, 0, 2L));

        pointStoreDouble.add(points.get(0).getValue(), 0);
        pointStoreDouble.add(points.get(1).getValue(), 1);
        tree.addPoint(0, points.get(0).getSequenceIndex());
        tree.addPoint(1, points.get(1).getSequenceIndex());
        assertNotEquals(pointStoreDouble.get(0)[0], pointStoreDouble.get(1)[0]);

        for (int i = 0; i < 10000; i++) {
            Weighted<double[]> point = points.get(i % points.size());
            tree.deletePoint(i % points.size(), point.getSequenceIndex());
            tree.addPoint(i % points.size(), point.getSequenceIndex());
        }
    }
}
