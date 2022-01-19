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
import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.validateInternalState;
import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

import java.util.Arrays;

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
 */
public class SmallNodeStore implements INodeStore {

    public static int MAX_SMALLNODESTORE_CAPACITY = Short.MAX_VALUE / 2;

    private final int capacity;
    private final short[] parentIndex;
    private final short[] leftIndex;
    private final short[] rightIndex;
    private final short[] cutDimension;
    private final float[] cutValue;
    private final short[] mass;
    private final int[] leafPointIndex;

    protected IndexManager freeNodeManager;
    protected IndexManager freeLeafManager;

    /**
     * Create a new NodeStore with the given capacity.
     * 
     * @param capacity The maximum number of Nodes whose data can be stored.
     */
    public SmallNodeStore(int capacity) {
        this.capacity = capacity;
        freeNodeManager = new IndexManager(capacity);
        freeLeafManager = new IndexManager(capacity + 1);
        parentIndex = new short[2 * capacity + 1];
        mass = new short[2 * capacity + 1];
        leftIndex = new short[capacity];
        rightIndex = new short[capacity];
        cutDimension = new short[capacity];
        cutValue = new float[capacity];
        leafPointIndex = new int[capacity + 1];
        Arrays.fill(parentIndex, (short) NULL);
        Arrays.fill(leftIndex, (short) NULL);
        Arrays.fill(rightIndex, (short) NULL);
        Arrays.fill(leafPointIndex, PointStore.INFEASIBLE_POINTSTORE_INDEX);
    }

    public SmallNodeStore(int capacity, int[] leftIndex, int[] rightIndex, short[] cutDimension, float[] cutValue,
            int[] leafMass, int[] leafPointIndex, int[] freeNodeIndexes, int freeNodeIndexPointer,
            int[] freeLeafIndexes, int freeLeafIndexPointer) {

        this.capacity = capacity;
        this.freeNodeManager = new IndexManager(capacity, freeNodeIndexes, freeNodeIndexPointer);
        this.freeLeafManager = new IndexManager(capacity + 1, freeLeafIndexes, freeLeafIndexPointer);
        this.leftIndex = convertToShort(leftIndex);
        this.rightIndex = convertToShort(rightIndex);
        this.parentIndex = deriveParentIndex(this.leftIndex, this.rightIndex);

        this.cutDimension = cutDimension;
        this.cutValue = cutValue;
        // copy leaf mass to the later half; if mass is not null
        if (leafMass != null) {
            validateInternalState(leafPointIndex != null, " incorrect state for needing samplers");
            this.mass = convertToShort(leafMass);
            this.leafPointIndex = leafPointIndex;
        } else {
            this.mass = new short[2 * capacity + 1];
            this.leafPointIndex = new int[capacity + 1];
            Arrays.fill(this.leafPointIndex, PointStore.INFEASIBLE_POINTSTORE_INDEX);
        }
        if (leafMass != null) {
            for (int i = 0; i < capacity; i++) {
                if (parentIndex[i] == NULL) {
                    rebuildMass(i);
                }
            }
        }
    }

    void rebuildMass(int node) {
        if (!isLeaf(node) && (leftIndex[node] != NULL && rightIndex[node] != NULL)) {
            rebuildMass(leftIndex[node]);
            rebuildMass(rightIndex[node]);
            mass[node] = (short) (mass[leftIndex[node]] + mass[rightIndex[node]]);
        }
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
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = (float) cutValue;
        this.cutDimension[index] = (short) cutDimension;
        this.leftIndex[index] = (short) leftIndex;
        this.rightIndex[index] = (short) rightIndex;
        this.parentIndex[index] = (short) parentIndex;
        this.mass[index] = (short) mass;
        return index;
    }

    public int addLeaf(int parentIndex, int pointIndex, int mass) {
        int index = freeLeafManager.takeIndex();
        this.parentIndex[index + capacity] = (short) parentIndex;
        this.mass[index + capacity] = (short) mass;
        this.leafPointIndex[index] = pointIndex;
        return index + capacity;
    }

    @Override
    public void setParentIndex(int index, int parent) {
        parentIndex[index] = (short) parent;
    }

    @Override
    public int getParentIndex(int index) {
        return parentIndex[index];
    }

    @Override
    public void delete(int index) {
        if (isLeaf(index)) {
            parentIndex[index] = NULL;
            leafPointIndex[computeLeafIndex(index)] = PointStore.INFEASIBLE_POINTSTORE_INDEX;
            mass[index] = 0;
            freeLeafManager.releaseIndex(computeLeafIndex(index));
        } else {
            mass[index] = 0;
            leftIndex[index] = NULL;
            rightIndex[index] = NULL;
            parentIndex[index] = NULL;
            freeNodeManager.releaseIndex(index);
        }
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

    public int[] getRightIndex() {
        return convertToInt(rightIndex);
    }

    @Override
    public void setRightIndex(int index, int child) {
        rightIndex[index] = (short) child;
    }

    @Override
    public int getLeftIndex(int index) {
        return leftIndex[index];
    }

    public int[] getLeftIndex() {
        return convertToInt(leftIndex);
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

    public int[] getCutDimension() {
        return convertToInt(cutDimension);
    }

    @Override
    public double getCutValue(int index) {
        return cutValue[index];
    }

    public double[] getCutValue() {
        return toDoubleArray(cutValue);
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
        while (index != NULL) {
            --mass[index];
            index = parentIndex[index];
        }
    }

    public int getSibling(int parent, int node) {
        return leftIndex[parent] == node ? rightIndex[parent] : leftIndex[parent];
    }

    short[] deriveParentIndex(short[] leftIndex, short[] rightIndex) {
        int capacity = leftIndex.length;
        checkState(rightIndex.length == capacity, "incorrect function call, arrays should be equal");
        short[] parentIndex = new short[2 * capacity + 1];
        Arrays.fill(parentIndex, (short) NULL);
        for (short i = 0; i < capacity; i++) {
            if (leftIndex[i] != NULL) {
                checkState(parentIndex[leftIndex[i]] == NULL, "incorrect state, conflicting parent");
                parentIndex[leftIndex[i]] = i;
            }
            if (rightIndex[i] != NULL) {
                checkState(parentIndex[rightIndex[i]] == NULL, "incorrect state, conflicting parent");
                parentIndex[rightIndex[i]] = i;
            }
        }
        return parentIndex;
    }

    @Override
    public boolean isLeaf(int index) {
        checkArgument(index >= 0, "index has to be non-negative");
        return computeLeafIndex(index) >= 0;
    }

    public int computeLeafIndex(int index) {
        return index - capacity;
    }

    @Override
    public int getPointIndex(int index) {
        return leafPointIndex[computeLeafIndex(index)];
    }

    @Override
    public int setPointIndex(int index, int pointIndex) {
        int newIndex = computeLeafIndex(index);
        int savedPointIndex = this.leafPointIndex[newIndex];
        this.leafPointIndex[newIndex] = pointIndex;
        return savedPointIndex;
    }

    public int[] getLeafFreeIndexes() {
        return freeLeafManager.getFreeIndexes();
    }

    public int[] getNodeFreeIndexes() {
        return freeNodeManager.getFreeIndexes();
    }

    public int[] getLeafPointIndex() {
        return leafPointIndex;
    }

    public int[] getLeafMass() {
        return convertToInt(mass);
    }

    public int getLeafFreeIndexPointer() {
        return freeLeafManager.getFreeIndexPointer();
    }

    public int getNodeFreeIndexPointer() {
        return freeNodeManager.getFreeIndexPointer();
    }

    public int getCapacity() {
        return freeNodeManager.getCapacity();
    }

    public int size() {
        return freeNodeManager.size();
    }

    public boolean isCanonicalAndNotALeaf() {
        boolean check = leftIndex.length == rightIndex.length;
        int leafCounter = leftIndex.length;
        int nodeCounter = 1;

        // the root = 0; which means node 0 has no parent and is in use
        check = check && (parentIndex[0] == -1) && freeNodeManager.occupied.get(0);
        for (int i = 0; i < size() && check; i++) {
            if (leftIndex[i] != NULL) {
                if (leftIndex[i] < leftIndex.length) {
                    check = (nodeCounter == leftIndex[i]);
                    ++nodeCounter;
                } else {
                    check = (leftIndex[i] == leafCounter);
                    ++leafCounter;
                }
                check = check && (rightIndex[i] != NULL);

                if (rightIndex[i] < rightIndex.length) {
                    check = check && (nodeCounter == rightIndex[i]);
                    ++nodeCounter;
                } else {
                    check = check && (rightIndex[i] == leafCounter);
                    ++leafCounter;
                }
            } else {
                check = check && (rightIndex[i] == NULL);
            }
        }
        return check;
    }

    short[] convertToShort(int[] original) {
        short[] newArray = new short[original.length];
        for (int i = 0; i < original.length; i++) {
            newArray[i] = (short) original[i];
        }
        return newArray;
    }

    int[] convertToInt(short[] original) {
        int[] newArray = new int[original.length];
        for (int i = 0; i < original.length; i++) {
            newArray[i] = original[i];
        }
        return newArray;
    }

}
