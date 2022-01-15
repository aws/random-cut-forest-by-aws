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

package com.amazon.randomcutforest.state.tree;

import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;
import static java.lang.Math.min;

import java.util.concurrent.ArrayBlockingQueue;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IContextualStateMapper;
import com.amazon.randomcutforest.state.Version;
import com.amazon.randomcutforest.state.store.NodeStoreState;
import com.amazon.randomcutforest.tree.AbstractNodeStore;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class AbstractNodeStoreMapper
        implements IContextualStateMapper<AbstractNodeStore, NodeStoreState, CompactRandomCutTreeContext> {

    private int root;

    @Override
    public AbstractNodeStore toModel(NodeStoreState state, CompactRandomCutTreeContext compactRandomCutTreeContext,
            long seed) {
        int capacity = state.getCapacity();
        int[] cutDimension = ArrayPacking.unpackInts(state.getCutDimension(), state.isCompressed());
        float[] cutValue = ArrayPacking.unpackFloats(state.getCutValueData());
        int[] leftIndex = ArrayPacking.unpackInts(state.getLeftIndex(), state.isCompressed());
        int[] rightIndex = ArrayPacking.unpackInts(state.getRightIndex(), state.isCompressed());
        if (state.isCanonicalAndNotALeaf()) {
            reverseBits(state.getSize(), leftIndex, rightIndex, capacity);
        } else {
            replaceLeaves(leftIndex, capacity);
            replaceLeaves(rightIndex, capacity);
        }

        // note boundingBoxCache is not set deliberately
        return AbstractNodeStore.builder().capacity(capacity).useRoot(root).leftIndex(leftIndex).rightIndex(rightIndex)
                .cutDimension(cutDimension).cutValues(cutValue)
                .dimensions(compactRandomCutTreeContext.getPointStore().getDimensions())
                .pointStoreView(compactRandomCutTreeContext.getPointStore()).build();
    }

    @Override
    public NodeStoreState toState(AbstractNodeStore model) {
        NodeStoreState state = new NodeStoreState();
        int capacity = model.getCapacity();
        state.setVersion(Version.V3_0);
        state.setCapacity(capacity);
        state.setCompressed(true);
        state.setPartialTreeStateEnabled(true);
        state.setPrecision(Precision.FLOAT_32.name());

        int[] leftIndex = model.getLeftIndex();
        int[] rightIndex = model.getRightIndex();
        int[] cutDimension = model.getCutDimension();
        float[] cutValues = model.getCutValues();

        int[] map = new int[capacity];
        int size = reorderNodesInBreadthFirstOrder(map, leftIndex, rightIndex, capacity);
        state.setSize(size);
        boolean check = root != Null && root < capacity;
        state.setCanonicalAndNotALeaf(check);
        if (check) { // can have a canonical representation saving a lot of space
            int[] reorderedLeftArray = new int[size];
            int[] reorderedRightArray = new int[size];
            int[] reorderedCutDimension = new int[size];
            float[] reorderedCutValue = new float[size];
            for (int i = 0; i < size; i++) {
                reorderedLeftArray[i] = (leftIndex[map[i]] < capacity) ? 1 : 0;
                reorderedRightArray[i] = (rightIndex[map[i]] < capacity) ? 1 : 0;
                reorderedCutDimension[i] = cutDimension[map[i]];
                reorderedCutValue[i] = cutValues[map[i]];
            }
            state.setLeftIndex(ArrayPacking.pack(reorderedLeftArray, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(reorderedRightArray, state.isCompressed()));
            state.setSize(model.size());
            state.setCutDimension(ArrayPacking.pack(reorderedCutDimension, state.isCompressed()));
            state.setCutValueData(ArrayPacking.pack(reorderedCutValue));
        }
        return state;
    }

    /**
     * The follong function takes a pair of left and right indices for a regular
     * binary tree (each node has 0 or 2 children) and where internal nodes are in
     * the range [0..capacity-1] the indices are represented as : 0 for internal
     * node; 1 for leaf node; the root is 0 and every non-leaf node is added to a
     * queue; the number assigned to that node is the number in the queue Note that
     * this implies that the left/right children can be represented by bit-arrays
     *
     * This function reflates the bits to the queue numbers
     *
     * @param size       the size of the two arrays, typically this is capacity; but
     *                   can be different in RCF2.0
     * @param leftIndex  the left bitarray
     * @param rightIndex the right bitarray
     * @param capacity   the number of internal nodes (one less than number of
     *                   leaves)
     */
    protected static void reverseBits(int size, int[] leftIndex, int[] rightIndex, int capacity) {
        int nodeCounter = 1;
        for (int i = 0; i < size; i++) {
            if (leftIndex[i] != 0) {
                leftIndex[i] = nodeCounter++;
            } else {
                leftIndex[i] = capacity;
            }
            if (rightIndex[i] != 0) {
                rightIndex[i] = nodeCounter++;
            } else {
                rightIndex[i] = capacity;
            }
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = capacity;
        }
    }

    /**
     * takes a non-negative array and truncates it (in place) to [-1..capacity]; the
     * intended use is that [0..capacity-1] are internal nodes and if the array
     * represents left/right indices then the reachable nodes with valu capacity
     * correspond to leaves (which would be differentiated outsde this mappaer)
     * 
     * @param array    input array
     * @param capacity the truncation value
     */
    protected static void replaceLeaves(int[] array, int capacity) {
        for (int i = 0; i < array.length; i++) {
            array[i] = min(array[i], capacity);
        }
    }

    /**
     * The following function reorders the nodes stored in the tree in a breadth
     * first order; Note that a regular binary tree where each internal node has 2
     * chidren, as is the case for AbstractRandomCutTree or any tree produced in a
     * Random Forest ensemble (not restricted to Random Cut Forests), has maxsize -
     * 1 internal nodes for maxSize number of leaves. The leaves are numbered 0 +
     * (maxsize), 1 + (maxSize), ..., etc. in that BFS ordering. The root is node 0.
     *
     * Note that if the binary tree is a complete binary tree, then the numbering
     * would correspond to the well known heuristic where children of node index i
     * are numbered 2*i and 2*i + 1. The trees in AbstractCompactRandomCutTree will
     * not be complete binary trees. But a similar numbering enables us to compress
     * the entire structure of the tree into two bit arrays corresponding to
     * presence of left and right children. The idea can be viewed as similar to
     * Zak's numbering for regular binary trees Lexicographic generation of binary
     * trees, S. Zaks, TCS volume 10, pages 63-82, 1980, that uses depth first
     * numbering. However an extensive literature exists on this topic.
     *
     * The overall relies on the extra advantage that we can use two bit sequences;
     * the left and right child pointers which appears to be simple. While it is
     * feasible to always maintain this order, that would complicate the standard
     * binary search tree pattern and this tranformation is used when the tree is
     * serialized. Note that while there is savings in representing the tree
     * structure into two bit arrays, the bulk of the serialization corresponds to
     * the payload at the nodes (cuts, dimensions for internal nodes and index to
     * pointstore, number of copies for the leaves). The translation to the bits is
     * handled by the NodeStoreMapper. The algorithm here corresponds to just
     * producing the cannoical order.
     *
     * The algorithm renumbers the nodes in BFS ordering.
     */
    public int reorderNodesInBreadthFirstOrder(int[] map, int[] leftIndex, int[] rightIndex, int capacity) {

        if ((root != Null) && (root < capacity)) {
            int currentNode = 0;
            ArrayBlockingQueue<Integer> nodeQueue = new ArrayBlockingQueue<>(capacity);
            nodeQueue.add(root);
            while (!nodeQueue.isEmpty()) {
                int head = nodeQueue.poll();
                int leftChild = leftIndex[head];
                if (leftChild < capacity) {
                    nodeQueue.add(leftChild);
                }
                int rightChild = rightIndex[head];
                if (rightChild < capacity) {
                    nodeQueue.add(rightChild);
                }
                map[currentNode] = head;
                currentNode++;
            }
            return currentNode;
        }
        return 0;
    }

}
