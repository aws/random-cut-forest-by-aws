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

package com.amazon.randomcutforest.state.store;

import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class NodeStoreMapper implements IStateMapper<NodeStore, NodeStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compress = true;

    /**
     * determines if a sampler is needed to bring up a tree
     */
    private boolean usePartialTrees = false;

    @Override
    public NodeStore toModel(NodeStoreState state, long seed) {
        int capacity = state.getCapacity();
        int[] cutDimension = ArrayPacking.unPackInts(state.getCutDimension(), state.isCompressed());
        double[] cutValue = Arrays.copyOf(state.getCutValueDouble(), state.getCutValueDouble().length);

        int[] leftIndex = ArrayPacking.unPackInts(state.getLeftIndex(), state.isCompressed());
        int[] rightIndex = ArrayPacking.unPackInts(state.getRightIndex(), state.isCompressed());
        if (state.isCanonicalAndNotALeaf()) {
            reverseBits(state.getSize(), leftIndex, rightIndex);
        }

        int[] leafMass;
        int[] leafPointIndex;
        if (state.isUsePartialTrees()) {
            leafMass = null;
            leafPointIndex = null;
        } else {
            leafMass = ArrayPacking.unPackInts(state.getLeafmass(), state.isCompressed());
            leafPointIndex = ArrayPacking.unPackInts(state.getLeafPointIndex(), state.isCompressed());
        }

        int[] nodeFreeIndexes = ArrayPacking.unPackInts(state.getNodeFreeIndexes(), state.isCompressed());
        int nodeFreeIndexPointer = state.getNodeFreeIndexPointer();
        int[] leafFreeIndexes = ArrayPacking.unPackInts(state.getLeafFreeIndexes(), state.isCompressed());
        int leafFreeIndexPointer = state.getLeafFreeIndexPointer();

        return new NodeStore(capacity, leftIndex, rightIndex, cutDimension, cutValue, leafMass, leafPointIndex,
                nodeFreeIndexes, nodeFreeIndexPointer, leafFreeIndexes, leafFreeIndexPointer);
    }

    @Override
    public NodeStoreState toState(NodeStore model) {
        NodeStoreState state = new NodeStoreState();
        state.setCapacity(model.getCapacity());
        state.setCompressed(compress);
        state.setUsePartialTrees(usePartialTrees);

        int[] leftIndex = Arrays.copyOf(model.leftIndex, model.leftIndex.length);
        int[] rightIndex = Arrays.copyOf(model.rightIndex, model.rightIndex.length);
        boolean check = state.isCompressed() && model.isCanonicalAndNotALeaf();
        state.setCanonicalAndNotALeaf(check);
        if (check) { // can have a canonical representation saving a lot of space
            reduceToBits(model.size(), leftIndex, rightIndex);
            state.setLeftIndex(ArrayPacking.pack(leftIndex, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(rightIndex, state.isCompressed()));
            state.setSize(model.size());
        } else { // the temporary array leftIndex and rightIndex may be corrupt in reduceToBits()
            state.setLeftIndex(ArrayPacking.pack(model.leftIndex, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(model.rightIndex, state.isCompressed()));
        }

        state.setCutDimension(ArrayPacking.pack(model.cutDimension, state.isCompressed()));
        state.setCutValueDouble(Arrays.copyOf(model.cutValue, model.cutValue.length));

        state.setNodeFreeIndexes(ArrayPacking.pack(model.getNodeFreeIndexes(), state.isCompressed()));
        state.setNodeFreeIndexPointer(model.getNodeFreeIndexPointer());
        state.setLeafFreeIndexes(ArrayPacking.pack(model.getLeafFreeIndexes(), state.isCompressed()));
        state.setLeafFreeIndexPointer(model.getLeafFreeIndexPointer());
        if (!state.isUsePartialTrees()) {
            state.setLeafPointIndex(ArrayPacking.pack(model.getLeafPointIndex(), state.isCompressed()));
            state.setLeafmass(ArrayPacking.pack(model.getLeafMass(), state.isCompressed()));
        }

        return state;
    }

    void reduceToBits(int size, int[] leftIndex, int[] rightIndex) {
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
    }

    void reverseBits(int size, int[] leftIndex, int[] rightIndex) {
        int nodeCounter = 1;
        int leafCounter = leftIndex.length;
        for (int i = 0; i < size; i++) {
            if (leftIndex[i] != 0) {
                leftIndex[i] = nodeCounter++;
            } else {
                leftIndex[i] = leafCounter++;
            }
            if (rightIndex[i] != 0) {
                rightIndex[i] = nodeCounter++;
            } else {
                rightIndex[i] = leafCounter++;
            }
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = NULL;
        }
    }
}
