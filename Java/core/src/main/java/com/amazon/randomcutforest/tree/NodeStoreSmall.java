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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toByteArray;
import static com.amazon.randomcutforest.CommonUtils.toCharArray;
import static com.amazon.randomcutforest.CommonUtils.toIntArray;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Stack;

import com.amazon.randomcutforest.store.IndexIntervalManager;

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
public class NodeStoreSmall extends AbstractNodeStore {

    private final byte[] parentIndex;
    private final char[] leftIndex;
    private final char[] rightIndex;
    public final byte[] cutDimension;
    private final byte[] mass;

    public NodeStoreSmall(AbstractNodeStore.Builder builder) {
        super(builder);
        mass = new byte[capacity];
        Arrays.fill(mass, (byte) 0);
        if (builder.storeParent) {
            parentIndex = new byte[capacity];
            Arrays.fill(parentIndex, (byte) capacity);
        } else {
            parentIndex = null;
        }
        if (builder.leftIndex == null) {
            leftIndex = new char[capacity];
            rightIndex = new char[capacity];
            cutDimension = new byte[capacity];
            Arrays.fill(leftIndex, (char) capacity);
            Arrays.fill(rightIndex, (char) capacity);
        } else {
            checkArgument(builder.leftIndex.length == capacity, " incorrect length");
            checkArgument(builder.rightIndex.length == capacity, " incorrect length");

            leftIndex = toCharArray(builder.leftIndex);
            rightIndex = toCharArray(builder.rightIndex);
            cutDimension = toByteArray(builder.cutDimension);
            BitSet bits = new BitSet(capacity);
            if (builder.root != Null) {
                bits.set(builder.root);
            }
            for (int i = 0; i < leftIndex.length; i++) {
                if (isInternal(leftIndex[i])) {
                    bits.set(leftIndex[i]);
                    if (parentIndex != null) {
                        parentIndex[leftIndex[i]] = (byte) i;
                    }
                }
            }
            for (int i = 0; i < rightIndex.length; i++) {
                if (isInternal(rightIndex[i])) {
                    bits.set(rightIndex[i]);
                    if (parentIndex != null) {
                        parentIndex[rightIndex[i]] = (byte) i;
                    }
                }
            }
            freeNodeManager = new IndexIntervalManager(capacity, capacity, bits);
            // need to set up parents using the root
        }
    }

    @Override
    public int addNode(Stack<int[]> pathToRoot, float[] point, long sequenceIndex, int pointIndex, int childIndex,
            int cutDimension, float cutValue, BoundingBoxFloat box) {
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = (byte) cutDimension;
        if (leftOf(cutValue, cutDimension, point)) {
            this.leftIndex[index] = (char) (pointIndex + capacity + 1);
            this.rightIndex[index] = (char) childIndex;
        } else {
            this.rightIndex[index] = (char) (pointIndex + capacity + 1);
            this.leftIndex[index] = (char) childIndex;
        }
        this.mass[index] = (byte) (((byte) getMass(childIndex) + 1) % (capacity + 1));
        addLeaf(pointIndex, sequenceIndex);
        int parentIndex = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
        if (this.parentIndex != null) {
            this.parentIndex[index] = (byte) parentIndex;
            if (!isLeaf(childIndex)) {
                this.parentIndex[childIndex] = (byte) (index);
            }
        }
        addBox(index, point, box);
        if (parentIndex != Null) {
            spliceEdge(parentIndex, childIndex, index);
            manageAncestorsAdd(pathToRoot, point, pointStoreView);
        }
        if (pointSum != null) {
            recomputePointSum(index);
        }
        return index;
    }

    @Override
    public void addToPartialTree(Stack<int[]> pathToRoot, float[] point, int pointIndex) {
        int node = pathToRoot.lastElement()[0];
        if (leftOf(node, point)) {
            leftIndex[node] = (char) (pointIndex + capacity + 1);
        } else {
            rightIndex[node] = (char) (pointIndex + capacity + 1);
        }
        manageAncestorsAdd(pathToRoot, point, pointStoreView);
    }

    public int getLeftIndex(int index) {
        return leftIndex[index];
    }

    public int getRightIndex(int index) {
        return rightIndex[index];
    }

    public void setRoot(int index) {
        if (!isLeaf(index) && parentIndex != null) {
            parentIndex[index] = (byte) capacity;
        }
    }

    @Override
    protected void decreaseMassOfInternalNode(int node) {
        mass[node] = (byte) (((mass[node] & 0xff) + capacity) % (capacity + 1)); // this cannot get to 0
    }

    @Override
    protected void increaseMassOfInternalNode(int node) {
        mass[node] = (byte) (((mass[node] & 0xff) + 1) % (capacity + 1));
        // mass of root == 0; note capacity = number_of_leaves - 1
    }

    public void deleteInternalNode(int index) {
        leftIndex[index] = (char) capacity;
        rightIndex[index] = (char) capacity;
        if (parentIndex != null) {
            parentIndex[index] = (byte) capacity;
        }
        if (pointSum != null) {
            invalidatePointSum(index);
        }
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            rangeSumData[idx] = 0.0;
        }
        freeNodeManager.releaseIndex(index);
    }

    public int getMass(int index) {
        return (isLeaf(index)) ? getLeafMass(index) : mass[index] != 0 ? (mass[index] & 0xff) : (capacity + 1);
    }

    public void spliceEdge(int parent, int node, int newNode) {
        assert (!isLeaf(newNode));
        if (node == leftIndex[parent]) {
            leftIndex[parent] = (char) newNode;
        } else {
            rightIndex[parent] = (char) newNode;
        }
        if (parentIndex != null && isInternal(node)) {
            parentIndex[node] = (byte) newNode;
        }
    }

    public void replaceParentBySibling(int grandParent, int parent, int node) {
        int sibling = getSibling(node, parent);
        if (parent == leftIndex[grandParent]) {
            leftIndex[grandParent] = (char) sibling;
        } else {
            rightIndex[grandParent] = (char) sibling;
        }
        if (parentIndex != null && isInternal(sibling)) {
            parentIndex[sibling] = (byte) grandParent;
        }
    }

    public int getCutDimension(int index) {
        return cutDimension[index] & 0xff;
    }

    public int[] getCutDimension() {
        return toIntArray(cutDimension);
    }

    public int[] getLeftIndex() {
        return toIntArray(leftIndex);
    }

    public int[] getRightIndex() {
        return toIntArray(rightIndex);
    }

}
