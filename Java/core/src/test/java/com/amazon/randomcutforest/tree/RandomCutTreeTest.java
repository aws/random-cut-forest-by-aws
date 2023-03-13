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

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.store.PointStore;

public class RandomCutTreeTest {

    private static final double EPSILON = 1e-8;

    private Random rng;
    private RandomCutTree tree;

    @BeforeEach
    public void setUp() {
        rng = mock(Random.class);
        PointStore pointStoreFloat = new PointStore.Builder().indexCapacity(100).capacity(100).initialSize(100)
                .dimensions(2).build();
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStoreFloat)
                .storeSequenceIndexesEnabled(true).dimension(2).build();

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

        assertThrows(IllegalArgumentException.class, () -> tree.setBoundingBoxCacheFraction(-0.5));
        assertThrows(IllegalArgumentException.class, () -> tree.setConfig("foo", 0));
        assertThrows(IllegalArgumentException.class, () -> tree.getConfig("bar"));
        assertEquals(tree.getConfig(Config.BOUNDING_BOX_CACHE_FRACTION), 1.0);
        tree.setConfig(Config.BOUNDING_BOX_CACHE_FRACTION, 0.2);

        assertEquals(pointStoreFloat.add(new float[] { -1, -1 }, 1), 0);
        assertEquals(pointStoreFloat.add(new float[] { 1, 1 }, 2), 1);
        assertEquals(pointStoreFloat.add(new float[] { -1, 0 }, 3), 2);
        assertEquals(pointStoreFloat.add(new float[] { 0, 1 }, 4), 3);
        assertEquals(pointStoreFloat.add(new float[] { 0, 1 }, 5), 4);
        assertEquals(pointStoreFloat.add(new float[] { 0, 0 }, 6), 5);

        assertThrows(IllegalStateException.class, () -> tree.deletePoint(0, 1));
        tree.addPoint(0, 1);

        when(rng.nextDouble()).thenReturn(0.625);
        tree.addPoint(1, 2);

        when(rng.nextDouble()).thenReturn(0.5);
        tree.addPoint(2, 3);

        when(rng.nextDouble()).thenReturn(0.25);
        tree.addPoint(3, 4);

        // add mass to 0,1
        tree.addPoint(4, 5);
    }

    /**
     * Verify that the tree has the form described in the setUp method.
     */
    @Test
    public void testInitialTreeState() {
        int node = tree.getRoot();
        // the second double[] is intentional
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(5));
        assertArrayEquals(new double[] { -1, 2 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);
        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(1L), 1);

        node = tree.nodeStore.getRightIndex(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new BoundingBox(new float[] { 1, 1 }));
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.nodeStore.getMass(node), is(4));
        assertArrayEquals(new double[] { 0.0, 3.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(2L), 1);

        node = tree.nodeStore.getLeftIndex(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));

        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.nodeStore.getMass(node), is(3));
        assertArrayEquals(new double[] { -1.0, 2.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);
        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(3L), 1);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(2));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(4L), 1);
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(5L), 1);
        assertThrows(IllegalStateException.class, () -> tree.deletePoint(5, 6));
    }

    @Test
    public void testDeletePointWithLeafSibling() {
        tree.deletePoint(2, 3);

        // root node bounding box and cut remains unchanged, mass and centerOfMass are
        // updated

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { 0.0, 2.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(1L), 1);
        // sibling node moves up and bounding box recomputed

        node = tree.nodeStore.getRightIndex(node);
        expectedBox = new BoundingBox(new float[] { 0, 1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.nodeStore.getMass(node), is(3));
        assertArrayEquals(new double[] { 1.0, 3.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(2));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(4L), 1);
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(5L), 1);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(2L), 1);
    }

    @Test
    public void testDeletePointWithNonLeafSibling() {
        tree.deletePoint(1, 2);

        // root node bounding box recomputed

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(1L), 1);

        // sibling node moves up and bounding box stays the same

        node = tree.nodeStore.getRightIndex(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(3L), 1);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(2));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(4L), 1);
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(5L), 1);
    }

    @Test
    public void testDeletePointWithMassGreaterThan1() {
        tree.deletePoint(3, 4);

        // same as initial state except mass at 0,1 is 1

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(1));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(1L), 1);
        assertArrayEquals(new double[] { -1.0, 1.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(1L), 1);

        node = tree.nodeStore.getRightIndex(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.nodeStore.getMass(node), is(3));

        assertArrayEquals(new double[] { 0.0, 2.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(2L), 1);

        node = tree.nodeStore.getLeftIndex(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.nodeStore.getBox(node), is(expectedBox));
        assertEquals(expectedBox.toString(), tree.nodeStore.getBox(node).toString());
        assertThat(tree.nodeStore.getCutDimension(node), is(0));
        assertThat(tree.nodeStore.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { -1.0, 1.0 }, toDoubleArray(tree.nodeStore.getPointSum(node)), EPSILON);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getLeftIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getLeftIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getLeftIndex(node))).get(3L), 1);

        assertThat(tree.nodeStore.isLeaf(tree.nodeStore.getRightIndex(node)), is(true));
        assertThat(tree.pointStoreView.get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.nodeStore.getMass(tree.nodeStore.getRightIndex(node)), is(1));
        assertEquals(tree.nodeStore.getSequenceMap()
                .get(tree.nodeStore.getPointIndex(tree.nodeStore.getRightIndex(node))).get(5L), 1);
    }

    @Test
    public void testDeletePointInvalid() {
        // specified sequence index does not exist
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(2, 99));

        // point does not exist in tree
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(7, 3));
    }

    @Test
    public void testUpdatesOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        PointStore pointStoreFloat = new PointStore.Builder().indexCapacity(10).capacity(10).currentStoreCapacity(10)
                .dimensions(1).build();
        RandomCutTree tree = RandomCutTree.builder().random(rng).pointStoreView(pointStoreFloat).build();

        List<Weighted<double[]>> points = new ArrayList<>();
        points.add(new Weighted<>(new double[] { 48.08 }, 0, 1L));
        points.add(new Weighted<>(new double[] { 48.08001 }, 0, 2L));

        pointStoreFloat.add(toFloatArray(points.get(0).getValue()), 0);
        pointStoreFloat.add(toFloatArray(points.get(1).getValue()), 1);
        tree.addPoint(0, points.get(0).getSequenceIndex());
        tree.addPoint(1, points.get(1).getSequenceIndex());
        assertNotEquals(pointStoreFloat.get(0)[0], pointStoreFloat.get(1)[0]);

        for (int i = 0; i < 10000; i++) {
            Weighted<double[]> point = points.get(i % points.size());
            tree.deletePoint(i % points.size(), point.getSequenceIndex());
            tree.addPoint(i % points.size(), point.getSequenceIndex());
        }
    }

    @Test
    public void testfloat() {
        float x = 110.13f;
        double sum = 0;
        int trials = 230000;
        for (int i = 0; i < trials; i++) {
            float z = (x * (trials - i + 1) - x);
            sum += z;
        }
        System.out.println(sum);
        for (int i = 0; i < trials - 1; i++) {
            float z = (x * (trials - i + 1) - x);
            sum -= z;
        }
        System.out.println(sum + " " + (double) x + " " + (sum <= (double) x));
        float[] possible = new float[trials];
        float[] alsoPossible = new float[trials];
        for (int i = 0; i < trials; i++) {
            possible[i] = x;
            alsoPossible[i] = (trials - i + 1) * x;
        }
        BoundingBox box = new BoundingBox(possible, alsoPossible);
        System.out.println("rangesum " + box.getRangeSum());
        double factor = 1.0 - 1e-16;
        System.out.println(factor);
        Cut cut = RandomCutTree.randomCut(factor, possible, box);
    }
}
