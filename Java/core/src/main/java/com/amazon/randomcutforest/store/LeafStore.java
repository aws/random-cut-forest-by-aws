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

package com.amazon.randomcutforest.store;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

import java.util.Arrays;

/**
 * A fixed-size buffer for storing leaf nodes. A leaf node is defined by its
 * parent node and its leaf node value. * The LeafStore class uses arrays to
 * store these field values for a collection of nodes. An index in the store can
 * be used to look up the field values for a particular leaf node.
 *
 * This handles leaf nodes which corresponds to [lowerRangeLimit .. max (short)]
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * leaf Nodes.
 *
 * The reference to the nodes will be offset by an amount that this ihe
 * capacity.
 */
public class LeafStore extends IndexManager implements ILeafStore {

    public final int[] pointIndex;
    public final int[] parentIndex;
    public final int[] mass;

    public LeafStore(int capacity) {
        super(capacity);
        pointIndex = new int[capacity];
        parentIndex = new int[capacity];
        mass = new int[capacity];
        Arrays.fill(pointIndex, PointStore.INFEASIBLE_INDEX);
        Arrays.fill(parentIndex, NULL);
    }

    public LeafStore(int[] pointIndex, int[] parentIndex, int[] mass, int[] freeIndexes, int freeIndexPointer) {
        super(freeIndexes, freeIndexPointer);

        checkNotNull(pointIndex, "pointIndex must not be null");
        checkNotNull(parentIndex, "parentIndex must not be null");
        checkNotNull(mass, "mass must not be null");

        int capacity = pointIndex.length;
        checkArgument(parentIndex.length == capacity && mass.length == capacity && freeIndexes.length == capacity,
                "all array arguments must have the same length");

        this.pointIndex = pointIndex;
        this.parentIndex = parentIndex;
        this.mass = mass;
    }

    public int addLeaf(int parentIndex, int pointIndex, int mass) {
        int index = takeIndex();
        this.parentIndex[index] = (short) parentIndex;
        this.mass[index] = (short) mass;
        this.pointIndex[index] = pointIndex;
        return index;
    }

    @Override
    public void setParent(int index, int parent) {
        parentIndex[index] = (short) parent;
    }

    @Override
    public int getParent(int index) {
        return parentIndex[index];
    }

    public void delete(int index) {
        releaseIndex(index);
    }

    @Override
    public int getPointIndex(int index) {
        return pointIndex[index];
    }

    @Override
    public int setPointIndex(int index, int pointIndex) {
        int savedIndex = this.pointIndex[index];
        this.pointIndex[index] = pointIndex;
        return savedIndex;
    }

    @Override
    public int incrementMass(int index) {
        return ++mass[index];
    }

    @Override
    public int decrementMass(int index) {
        return --mass[index];
    }

    @Override
    public int getMass(int index) {
        return mass[index];
    }
}
