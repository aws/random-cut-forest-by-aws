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

import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.SmallNodeStore;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class SmallNodeStoreMapper implements IStateMapper<SmallNodeStore, NodeStoreState> {

    /**
     * if single precision, then stores the cut information as a float array
     * (converted to bytes)
     */
    private Precision precision = Precision.FLOAT_64;

    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    /**
     * determines if a sampler is needed to bring up a tree
     */
    private boolean partialTreeStateEnabled = false;

    @Override
    public SmallNodeStore toModel(NodeStoreState state, long seed) {
        checkState(state.getPrecisionEnumValue() == Precision.FLOAT_32, " incorrect invocation of SmallNodeStore");
        int capacity = state.getCapacity();
        short[] cutDimension = ArrayPacking.unpackShorts(state.getCutDimension(), state.isCompressed());
        float[] cutValue = ArrayPacking.unpackFloats(state.getCutValueData());
        short[] leftIndex = ArrayPacking.unpackShorts(state.getLeftIndex(), state.isCompressed());
        short[] rightIndex = ArrayPacking.unpackShorts(state.getRightIndex(), state.isCompressed());
        if (state.isCanonicalAndNotALeaf()) {
            reverseBits(state.getSize(), leftIndex, rightIndex);
        }

        short[] leafMass;
        int[] leafPointIndex;
        if (state.isPartialTreeStateEnabled()) {
            leafMass = null;
            leafPointIndex = null;
        } else {
            leafMass = ArrayPacking.unpackShorts(state.getLeafMass(), state.isCompressed());
            leafPointIndex = ArrayPacking.unpackInts(state.getLeafPointIndex(), state.isCompressed());
        }

        short[] nodeFreeIndexes = ArrayPacking.unpackShorts(state.getNodeFreeIndexes(), state.isCompressed());
        int nodeFreeIndexPointer = state.getNodeFreeIndexPointer();
        short[] leafFreeIndexes = ArrayPacking.unpackShorts(state.getLeafFreeIndexes(), state.isCompressed());
        int leafFreeIndexPointer = state.getLeafFreeIndexPointer();

        return new SmallNodeStore(capacity, leftIndex, rightIndex, cutDimension, cutValue, leafMass, leafPointIndex,
                nodeFreeIndexes, nodeFreeIndexPointer, leafFreeIndexes, leafFreeIndexPointer);
    }

    @Override
    public NodeStoreState toState(SmallNodeStore model) {
        NodeStoreState state = new NodeStoreState();
        state.setCapacity(model.getCapacity());
        state.setCompressed(compressionEnabled);
        state.setPartialTreeStateEnabled(partialTreeStateEnabled);
        state.setPrecision(precision.name());

        int[] leftIndex = Arrays.copyOf(model.getLeftIndex(), model.getLeftIndex().length);
        int[] rightIndex = Arrays.copyOf(model.getRightIndex(), model.getRightIndex().length);
        boolean check = state.isCompressed() && model.isCanonicalAndNotALeaf();
        state.setCanonicalAndNotALeaf(check);
        if (check) { // can have a canonical representation saving a lot of space
            NodeStoreMapper.reduceToBits(model.size(), leftIndex, rightIndex);
            state.setLeftIndex(ArrayPacking.pack(leftIndex, state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(rightIndex, state.isCompressed()));
            state.setSize(model.size());
        } else { // the temporary array leftIndex and rightIndex may be corrupt in reduceToBits()
            state.setLeftIndex(ArrayPacking.pack(model.getLeftIndex(), state.isCompressed()));
            state.setRightIndex(ArrayPacking.pack(model.getRightIndex(), state.isCompressed()));
        }

        state.setCutDimension(ArrayPacking.pack(model.getCutDimension(), state.isCompressed()));
        state.setCutValueData(ArrayPacking.pack(CommonUtils.toFloatArray(model.getCutValue())));
        state.setNodeFreeIndexes(ArrayPacking.pack(model.getNodeFreeIndexes(), state.isCompressed()));
        state.setNodeFreeIndexPointer(model.getNodeFreeIndexPointer());
        state.setLeafFreeIndexes(ArrayPacking.pack(model.getLeafFreeIndexes(), state.isCompressed()));
        state.setLeafFreeIndexPointer(model.getLeafFreeIndexPointer());
        if (!state.isPartialTreeStateEnabled()) {
            state.setLeafPointIndex(ArrayPacking.pack(model.getLeafPointIndex(), state.isCompressed()));
            state.setLeafMass(ArrayPacking.pack(model.getLeafMass(), state.isCompressed()));
        }

        return state;
    }

    void reverseBits(int size, short[] leftIndex, short[] rightIndex) {
        short nodeCounter = 1;
        short leafCounter = (short) leftIndex.length;
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
            leftIndex[i] = rightIndex[i] = (short) NULL;
        }
    }
}
