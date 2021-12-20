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
 * Note that a NodeStore does not store instances of the
 * {@link com.amazon.randomcutforest.tree.Node} class.
 */
public class NodeStoreLarge extends AbstractNodeStore {

    private final int[] parentIndex;
    private final int[] leftIndex;
    private final int[] rightIndex;
    public final int[] cutDimension;
    private final int[] mass;

    /**
     * Create a new NodeStore with the given capacity.
     *
     * @param capacity The maximum number of Nodes whose data can be stored.
     */
    public NodeStoreLarge(int capacity, int dimensions, double nodeCacheFraction) {
        super(capacity, dimensions, nodeCacheFraction);
        mass = new int[capacity - 1];
        Arrays.fill(mass, (char) 0);
        if (nodeCacheFraction > 0) {
            parentIndex = new int[capacity - 1];
            Arrays.fill(parentIndex, (char) 0);
        } else {
            parentIndex = null;
        }
        leftIndex = new int[capacity - 1];
        rightIndex = new int[capacity - 1];
        cutDimension = new int[capacity - 1];
        Arrays.fill(leftIndex, 0);
        Arrays.fill(rightIndex, 0);
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
    @Override
    public int addNode(int parentIndex, int leftIndex, int rightIndex, int cutDimension, double cutValue, int mass) {
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = (float) cutValue;
        this.cutDimension[index] = cutDimension;
        this.leftIndex[index] = leftIndex;
        this.rightIndex[index] = rightIndex;
        this.mass[index] = (char) (mass - 1);
        if (this.parentIndex != null) {
            this.parentIndex[index] = parentIndex;
            if (!isLeaf(leftIndex)) {
                this.parentIndex[leftIndex - 1] = (index + 1);
            }
            if (!isLeaf(rightIndex)) {
                this.parentIndex[rightIndex - 1] = (index + 1);
            }
        }
        return index + 1;
    }

    public int getLeftIndex(int index) {
        return leftIndex[index - 1];
    }

    public int getRightIndex(int index) {
        return rightIndex[index - 1];
    }

    public void setRoot(int index) {
        if (!isLeaf(index) && parentIndex != null) {
            parentIndex[index - 1] = 0;
        }
    }

    @Override
    protected void decreaseMassOfInternalNode(int node) {
        --mass[node - 1];
    }

    @Override
    protected void increaseMassOfInternalNode(int node) {
        ++mass[node - 1];
    }

    public void deleteInternalNode(int index) {
        leftIndex[index - 1] = 0;
        rightIndex[index - 1] = 0;
        mass[index - 1] = 0;
        if (parentIndex != null) {
            parentIndex[index - 1] = 0;
        }
        cutDimension[index - 1] = Integer.MAX_VALUE;
        cutValue[index - 1] = (float) 0.0;
        freeNodeManager.releaseIndex(index - 1);
    }

    public int getMass(int index) {
        return (isLeaf(index)) ? getLeafMass(index) : (mass[index - 1] + 1);
    }

    public void spliceEdge(int parent, int node, int newNode) {
        assert (!isLeaf(newNode));
        if (node == leftIndex[parent - 1]) {
            leftIndex[parent - 1] = newNode;
        } else {
            rightIndex[parent - 1] = newNode;
        }
        if (!isLeaf(node) && nodeCacheFraction > 0.0) {
            parentIndex[node - 1] = newNode;
        }
    }

    public void replaceParentBySibling(int grandParent, int parent, int node) {
        int sibling = getSibling(node, parent);
        if (parent == leftIndex[grandParent - 1]) {
            leftIndex[grandParent - 1] = sibling;
        } else {
            rightIndex[grandParent - 1] = sibling;
        }
        if (!isLeaf(sibling) && nodeCacheFraction > 0.0) {
            parentIndex[sibling - 1] = grandParent;
        }
    }

    public int getCutDimension(int index) {
        return cutDimension[index - 1];
    }

}
