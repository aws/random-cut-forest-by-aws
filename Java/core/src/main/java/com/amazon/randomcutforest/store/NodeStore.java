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

/**
 * A fixed-size buffer for storing interior tree nodes. An interior node is
 * defined by its location in the tree (parent and child nodes), its random cut,
 * and its bounding box. The NodeStore class uses arrays to store these field
 * values for a collection of nodes. An index in the store can be used to look
 * up the field values for a particular node.
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * Nodes.
 *
 * Note that a NodeStore does not store instances of the
 * {@link com.amazon.randomcutforest.tree.Node} class.
 */
public class NodeStore extends SmallIndexManager {

    public final short[] parentIndex;
    public final short[] leftIndex;
    public final short[] rightIndex;
    public final int[] cutDimension;
    public final double[] cutValue;
    public final int[] mass;
    // public final BoundingBox[] boundingBox;

    /**
     * Create a new NodeStore with the given capacity.
     * 
     * @param capacity The maximum number of Nodes whose data can be stored.
     */
    public NodeStore(short capacity) {
        super(capacity);
        parentIndex = new short[capacity];
        leftIndex = new short[capacity];
        rightIndex = new short[capacity];
        cutDimension = new int[capacity];
        cutValue = new double[capacity];
        mass = new int[capacity];
    }

    public NodeStore(short[] parentIndex, short[] leftIndex, short[] rightIndex, int[] cutDimension, double[] cutValue,
            int[] mass, short[] freeIndexes, short freeIndexPointer) {
        // TODO validations
        super(freeIndexes, freeIndexPointer);
        this.parentIndex = parentIndex;
        this.leftIndex = leftIndex;
        this.rightIndex = rightIndex;
        this.cutDimension = cutDimension;
        this.cutValue = cutValue;
        this.mass = mass;
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
    public short addNode(short parentIndex, short leftIndex, short rightIndex, int cutDimension, double cutValue,
            int mass) {
        short index = takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = cutDimension;
        this.leftIndex[index] = leftIndex;
        this.rightIndex[index] = rightIndex;
        this.parentIndex[index] = parentIndex;
        this.mass[index] = mass;
        return index;
    }

    public void delete(short index) {
        releaseIndex(index);
    }

}
