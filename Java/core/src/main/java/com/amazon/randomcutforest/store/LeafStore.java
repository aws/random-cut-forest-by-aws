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
 * A fixed-size buffer for storing leaf nodes. A leaf node is defined by its
 * parent node and its leaf node value. * The LeafStore class uses arrays to
 * store these field values for a collection of nodes. An index in the store can
 * be used to look up the field values for a particular leaf node.
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * leaf Nodes.
 */
public class LeafStore extends SmallStoreCommon {

    public final int[] pointIndex;
    public final short[] parentIndex;
    public final int[] mass;

    public LeafStore(short capacity) {
        super(capacity);
        pointIndex = new int[capacity];
        parentIndex = new short[capacity];
        mass = new int[capacity];
    }

    public short addLeaf(short parentIndex, int pointIndex, int mass) {
        short index = takeIndex();
        this.parentIndex[index] = parentIndex;
        this.mass[index] = mass;
        this.pointIndex[index] = pointIndex;
        return index;
    }

    public void removeLeaf(short index) {
        releaseIndex(index);
    }
}
