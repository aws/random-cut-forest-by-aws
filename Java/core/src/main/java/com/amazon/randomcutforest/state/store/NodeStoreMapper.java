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

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.util.ArrayPacking;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

@Getter
@Setter
public class NodeStoreMapper implements IStateMapper<NodeStore, NodeStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public NodeStore toModel(NodeStoreState state, long seed) {
        int capacity = state.getCapacity();
        int[] cutDimension = ArrayPacking.unPackInts(state.getCutDimension(), state.isCompressed());
        int[] leftIndex = ArrayPacking.unPackInts(state.getLeftIndex(), state.isCompressed());
        int[] rightIndex = ArrayPacking.unPackInts(state.getRightIndex(), state.isCompressed());
        int[] leafMass = ArrayPacking.unPackInts(state.getLeafmass(), state.isCompressed());
        int[] leafPointIndex = ArrayPacking.unPackInts(state.getLeafPointIndex(), state.isCompressed());
        int[] nodeFreeIndexes = ArrayPacking.unPackInts(state.getNodeFreeIndexes(), state.isCompressed());
        int nodeFreeIndexPointer = state.getNodeFreeIndexPointer();
        int[] leafFreeIndexes = ArrayPacking.unPackInts(state.getLeafFreeIndexes(), state.isCompressed());
        int leafFreeIndexPointer = state.getLeafFreeIndexPointer();
        double[] cutValue = Arrays.copyOf(state.getCutValueDouble(), state.getCutValueDouble().length);

        return new NodeStore(capacity, leftIndex, rightIndex, cutDimension, cutValue, leafMass, leafPointIndex,
                nodeFreeIndexes, nodeFreeIndexPointer, leafFreeIndexes, leafFreeIndexPointer);
    }

    @Override
    public NodeStoreState toState(NodeStore model) {
        NodeStoreState state = new NodeStoreState();
        state.setCompressed(true);
        state.setCapacity(model.getCapacity());
        state.setLeftIndex(ArrayPacking.pack(model.leftIndex, state.isCompressed()));
        state.setRightIndex(ArrayPacking.pack(model.rightIndex, state.isCompressed()));
        state.setCutDimension(ArrayPacking.pack(model.cutDimension, state.isCompressed()));
        state.setNodeFreeIndexes(ArrayPacking.pack(model.getNodeFreeIndexes(), state.isCompressed()));
        state.setNodeFreeIndexPointer(model.getNodeFreeIndexPointer());
        state.setLeafFreeIndexes(ArrayPacking.pack(model.getLeafFreeIndexes(), state.isCompressed()));
        state.setLeafFreeIndexPointer(model.getLeafFreeIndexPointer());
        state.setLeafPointIndex(ArrayPacking.pack(model.getLeafPointIndex(), state.isCompressed()));
        state.setLeafmass(ArrayPacking.pack(model.getLeafMass(), state.isCompressed()));
        state.setCutValueDouble(Arrays.copyOf(model.cutValue, model.cutValue.length));
        return state;
    }
}
