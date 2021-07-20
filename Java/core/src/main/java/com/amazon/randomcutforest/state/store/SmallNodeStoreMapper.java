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
     * SmallNodeStore should only be applied to FLOAT_32
     */
    private Precision precision = Precision.FLOAT_32;

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
        int[] leftIndex = ArrayPacking.unpackInts(state.getLeftIndex(), state.isCompressed());
        int[] rightIndex = ArrayPacking.unpackInts(state.getRightIndex(), state.isCompressed());
        if (state.isCanonicalAndNotALeaf()) {
            NodeStoreMapper.reverseBits(state.getSize(), leftIndex, rightIndex);
        }

        int[] leafMass;
        int[] leafPointIndex;
        if (state.isPartialTreeStateEnabled()) {
            leafMass = null;
            leafPointIndex = null;
        } else {
            leafMass = ArrayPacking.unpackInts(state.getLeafMass(), state.isCompressed());
            leafPointIndex = ArrayPacking.unpackInts(state.getLeafPointIndex(), state.isCompressed());
        }

        int[] nodeFreeIndexes = ArrayPacking.unpackInts(state.getNodeFreeIndexes(), state.isCompressed());
        int nodeFreeIndexPointer = state.getNodeFreeIndexPointer();
        int[] leafFreeIndexes = ArrayPacking.unpackInts(state.getLeafFreeIndexes(), state.isCompressed());
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

}
