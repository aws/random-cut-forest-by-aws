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

import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A fixed-size buffer for storing interior tree nodes. An interior node is
 * defined by its location in the tree (parent and child nodes), its random cut,
 * and its bounding box. The NodeStore class uses arrays to store these field
 * values for a collection of nodes. An index in the store can be used to look
 * up the field values for a particular node.
 *
 * The internal nodes (handled by this store) corresponds to
 * [0..upperRangeLimit]
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * Nodes.
 *
 * Note that a NodeStore does not store instances of the
 * {@link com.amazon.randomcutforest.tree.Node} class.
 */
public class SmallNodeStore extends IndexManager implements INodeStore {

    public static final int MAX_TREE_SIZE = 16383;

    public final short[] parentIndex;
    public final short[] leftIndex;
    public final short[] rightIndex;
    public final int[] cutDimension;
    public final double[] cutValue;
    public final short[] mass;

    /**
     * Create a new NodeStore with the given capacity.
     * 
     * @param capacity The maximum number of Nodes whose data can be stored.
     */
    public SmallNodeStore(short capacity) {
        super(capacity);
        parentIndex = new short[capacity];
        leftIndex = new short[capacity];
        rightIndex = new short[capacity];
        cutDimension = new int[capacity];
        cutValue = new double[capacity];
        mass = new short[capacity];
        Arrays.fill(parentIndex, (short) NULL);
        Arrays.fill(leftIndex, (short) NULL);
        Arrays.fill(rightIndex, (short) NULL);
    }

    public SmallNodeStore(int capacity, short[] leftIndex, short[] rightIndex, int[] cutDimension, double[] cutValue,
            int[] freeIndexes, int freeIndexPointer) {
        // TODO validations
        super(capacity, freeIndexes, freeIndexPointer);
        this.parentIndex = getParentIndex(leftIndex, rightIndex);
        this.leftIndex = leftIndex;
        this.rightIndex = rightIndex;
        this.cutDimension = cutDimension;
        this.cutValue = cutValue;
        this.mass = new short[capacity];
    }

    /**
     * Add new node data to this store.
     * 
     * @param mass         Node mass.
     * @param parentIndex  Index of the parent node.
     * @param leftIndex    Index of the left child node.
     * @param rightIndex   Index of the right child node.
     * @param cutDimension The dimension of the cut in this node.
     * @param cutValue     The value of the cut in this node.
     * @return the index of the newly stored node.
     */
    public int addNode(int parentIndex, int leftIndex, int rightIndex, int cutDimension, double cutValue, int mass) {
        int index = takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = cutDimension;
        this.leftIndex[index] = (short) leftIndex;
        this.rightIndex[index] = (short) rightIndex;
        this.parentIndex[index] = (short) parentIndex;
        this.mass[index] = (short) mass;
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

    @Override
    public void delete(int index) {
        releaseIndex(index);
    }

    @Override
    public void replaceChild(int parent, int oldIndex, int newIndex) {
        if (leftIndex[parent] == oldIndex) {
            leftIndex[parent] = (short) newIndex;
        } else {
            rightIndex[parent] = (short) newIndex;
        }
    }

    @Override
    public int getRightIndex(int index) {
        return rightIndex[index];
    }

    @Override
    public void setRightIndex(int index, int child) {
        rightIndex[index] = (short) child;
    }

    @Override
    public int getLeftIndex(int index) {
        return leftIndex[index];
    }

    @Override
    public void setLeftIndex(int index, int child) {
        leftIndex[index] = (short) child;
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
    public int getCutDimension(int index) {
        return cutDimension[index];
    }

    @Override
    public double getCutValue(int index) {
        return cutValue[index];
    }

    @Override
    public int getMass(int index) {
        return mass[index];
    }

    @Override
    public void setMass(int index, int newMass) {
        mass[index] = (short) newMass;
    }

    @Override
    public void increaseMassOfSelfAndAncestors(int index) {
        while (index != NULL) {
            ++mass[index];
            index = parentIndex[index];
        }
    }

    @Override
    public void decreaseMassOfSelfAndAncestors(int index) {
        while (index != (short) NULL) {
            --mass[index];
            index = parentIndex[index];
        }
    }

    @Override
    public int getSibling(int parent, int node) {
        return leftIndex[parent] == (short) node ? rightIndex[parent] : leftIndex[parent];
    }

    @Override
    public Map<Integer, Integer> getLeavesAndParents() {
        HashMap<Integer, Integer> newMap = new HashMap<>();
        for (int i = 0; i < capacity; i++) {
            if (occupied.get(i) && leftIndex[i] >= capacity) {
                newMap.put((int) leftIndex[i], i);
            }
            if (occupied.get(i) && rightIndex[i] >= capacity) {
                newMap.put((int) rightIndex[i], i);
            }
        }
        return newMap;
    }

    /**
     * this function will help in reducing redundant information for the
     * constructors
     * 
     * @param leftIndex  the left child array
     * @param rightIndex the right child array
     * @return an array identifying the parent ( -1 or NULL for non-existent
     *         parents)
     */
    short[] getParentIndex(short[] leftIndex, short[] rightIndex) {
        int capacity = leftIndex.length;
        checkState(rightIndex.length == capacity, "incorrect function call, arrays should be equal");
        short[] parentIndex = new short[capacity];
        Arrays.fill(parentIndex, (short) NULL);
        for (short i = 0; i < capacity; i++) {
            if (leftIndex[i] != (short) NULL && leftIndex[i] < capacity) {
                checkState(parentIndex[leftIndex[i]] == (short) NULL, "incorrect state, conflicting parent");
                parentIndex[leftIndex[i]] = i;
            }
            if (rightIndex[i] != (short) NULL && rightIndex[i] < capacity) {
                checkState(parentIndex[rightIndex[i]] == (short) NULL, "incorrect state, conflicting parent");
                parentIndex[rightIndex[i]] = i;
            }
        }
        return parentIndex;
    }

}
