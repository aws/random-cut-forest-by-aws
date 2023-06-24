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
 */
public abstract class AbstractNodeStore {

    public static int Null = -1;

    public static boolean DEFAULT_STORE_PARENT = false;

    /**
     * the number of internal nodes; the nodes will range from 0..capacity-1 the
     * value capacity would correspond to "not yet set" the values Y= capacity+1+X
     * correspond to pointstore index X note that capacity + 1 + X =
     * number_of_leaves + X
     */
    protected final int capacity;
    protected final float[] cutValue;
    protected IndexIntervalManager freeNodeManager;

    public AbstractNodeStore(AbstractNodeStore.Builder<?> builder) {
        this.capacity = builder.capacity;
        if ((builder.leftIndex == null)) {
            freeNodeManager = new IndexIntervalManager(capacity);
        }
        cutValue = (builder.cutValues != null) ? builder.cutValues : new float[capacity];
    }

    protected abstract int addNode(Stack<int[]> pathToRoot, float[] point, long sendex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cutValue, BoundingBox box);

    public boolean isLeaf(int index) {
        return index > capacity;
    }

    public boolean isInternal(int index) {
        return index < capacity && index >= 0;
    }

    public abstract void assignInPartialTree(int savedParent, float[] point, int childReference);

    public abstract int getLeftIndex(int index);

    public abstract int getRightIndex(int index);

    public abstract int getParentIndex(int index);

    public abstract void setRoot(int index);

    protected abstract void decreaseMassOfInternalNode(int node);

    protected abstract void increaseMassOfInternalNode(int node);

    protected void manageInternalNodesPartial(Stack<int[]> path) {
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            increaseMassOfInternalNode(index);
        }
    }

    public Stack<int[]> getPath(int root, float[] point, boolean verbose) {
        int node = root;
        Stack<int[]> answer = new Stack<>();
        answer.push(new int[] { root, capacity });
        while (isInternal(node)) {
            double y = getCutValue(node);
            if (leftOf(node, point)) {
                answer.push(new int[] { getLeftIndex(node), getRightIndex(node) });
                node = getLeftIndex(node);
            } else { // this would push potential Null, of node == capacity
                     // that would be used for tree reconstruction
                answer.push(new int[] { getRightIndex(node), getLeftIndex(node) });
                node = getRightIndex(node);
            }
        }
        return answer;
    }

    public abstract void deleteInternalNode(int index);

    public abstract int getMass(int index);

    protected boolean leftOf(float cutValue, int cutDimension, float[] point) {
        return point[cutDimension] <= cutValue;
    }

    public boolean leftOf(int node, float[] point) {
        int cutDimension = getCutDimension(node);
        return leftOf(cutValue[node], cutDimension, point);
    }

    public int getSibling(int node, int parent) {
        int sibling = getLeftIndex(parent);
        if (node == sibling) {
            sibling = getRightIndex(parent);
        }
        return sibling;
    }

    public abstract void spliceEdge(int parent, int node, int newNode);

    public abstract void replaceParentBySibling(int grandParent, int parent, int node);

    public abstract int getCutDimension(int index);

    public double getCutValue(int index) {
        return cutValue[index];
    }

    protected boolean toLeft(float[] point, int currentNodeOffset) {
        return point[getCutDimension(currentNodeOffset)] <= cutValue[currentNodeOffset];
    }

    public abstract int[] getCutDimension();

    public abstract int[] getRightIndex();

    public abstract int[] getLeftIndex();

    public float[] getCutValues() {
        return cutValue;
    }

    public int getCapacity() {
        return capacity;
    }

    public int size() {
        return capacity - freeNodeManager.size();
    }

    /**
     * a builder
     */

    public static class Builder<T extends Builder<T>> {
        protected int capacity;
        protected int[] leftIndex;
        protected int[] rightIndex;
        protected int[] cutDimension;
        protected float[] cutValues;
        protected boolean storeParent = DEFAULT_STORE_PARENT;
        protected int dimension;
        protected int root;

        // maximum number of points in the store
        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T dimension(int dimension) {
            this.dimension = dimension;
            return (T) this;
        }

        public T useRoot(int root) {
            this.root = root;
            return (T) this;
        }

        public T leftIndex(int[] leftIndex) {
            this.leftIndex = leftIndex;
            return (T) this;
        }

        public T rightIndex(int[] rightIndex) {
            this.rightIndex = rightIndex;
            return (T) this;
        }

        public T cutDimension(int[] cutDimension) {
            this.cutDimension = cutDimension;
            return (T) this;
        }

        public T cutValues(float[] cutValues) {
            this.cutValues = cutValues;
            return (T) this;
        }

        public T storeParent(boolean storeParent) {
            this.storeParent = storeParent;
            return (T) this;
        }

        public AbstractNodeStore build() {
            if (leftIndex == null) {
                checkArgument(rightIndex == null, " incorrect option of right indices");
                checkArgument(cutValues == null, "incorrect option of cut values");
                checkArgument(cutDimension == null, " incorrect option of cut dimensions");
            } else {
                checkArgument(rightIndex.length == capacity, " incorrect length of right indices");
                checkArgument(cutValues.length == capacity, "incorrect length of cut values");
                checkArgument(cutDimension.length == capacity, " incorrect length of cut dimensions");
            }

            // capacity is numbner of internal nodes
            if (capacity < 256 && dimension <= 256) {
                return new NodeStoreSmall(this);
            } else if (capacity < Character.MAX_VALUE && dimension <= Character.MAX_VALUE) {
                return new NodeStoreMedium(this);
            } else {
                return new NodeStoreLarge(this);
            }
        }

    }

    public static Builder builder() {
        return new Builder();
    }

}
