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

import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;
import static java.lang.Math.min;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IContextualStateMapper;
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

        int[] nodeFreeIndexes = ArrayPacking.unpackInts(state.getNodeFreeIndexes(), state.isCompressed());
        return AbstractNodeStore.builder().capacity(capacity).useRoot(root).leftIndex(leftIndex).rightIndex(rightIndex)
                .cutDimension(cutDimension).cutValues(cutValue).freeIndicesIntervalArray(nodeFreeIndexes)
                .pointStoreView(compactRandomCutTreeContext.getPointStore()).build();
    }

    @Override
    public NodeStoreState toState(AbstractNodeStore model) {
        NodeStoreState state = new NodeStoreState();
        int capacity = model.getCapacity();
        state.setCapacity(capacity);
        state.setCompressed(true);
        state.setPartialTreeStateEnabled(true);
        state.setPrecision(Precision.FLOAT_32.name());

        int[] leftIndex = Arrays.copyOf(model.getLeftIndex(), model.getLeftIndex().length);
        int[] rightIndex = Arrays.copyOf(model.getRightIndex(), model.getRightIndex().length);
        boolean check = state.isCompressed() && model.isCanonicalAndNotALeaf();
        state.setCanonicalAndNotALeaf(check);
        if (check) { // can have a canonical representation saving a lot of space
            reduceToBits(model.size(), leftIndex, rightIndex);
            state.setLeftIndex(ArrayPacking.pack(leftIndex, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(rightIndex, state.isCompressed()));
            state.setSize(model.size());
        } else { // the temporary array leftIndex and rightIndex may be corrupt in reduceToBits()
            int[] left = model.getLeftIndex();
            replaceLeaves(left, capacity);
            int[] right = model.getRightIndex();
            replaceLeaves(right, capacity);
            state.setLeftIndex(ArrayPacking.pack(left, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(right, state.isCompressed()));
        }

        state.setCutDimension(ArrayPacking.pack(model.getCutDimension(), state.isCompressed()));
        state.setCutValueData(ArrayPacking.pack(model.getCutValue()));
        state.setNodeFreeIndexes(ArrayPacking.pack(model.getNodeFreeIndexes(), state.isCompressed()));

        return state;
    }

    protected static void reduceToBits(int size, int[] leftIndex, int[] rightIndex) {
        for (int i = 0; i < size; i++) {
            if (leftIndex[i] != NULL) {
                if (leftIndex[i] < leftIndex.length) {
                    leftIndex[i] = 1;
                } else {
                    leftIndex[i] = 0;
                }

                if (rightIndex[i] < rightIndex.length) {
                    rightIndex[i] = 1;
                } else {
                    rightIndex[i] = 0;
                }
            }
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = 0;
        }
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
}
