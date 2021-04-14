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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.SmallNodeStore;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class SmallNodeStoreMapper implements IStateMapper<SmallNodeStore, NodeStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public SmallNodeStore toModel(NodeStoreState state, long aaa) {
        int capacity = state.getCapacity();
        short[] leftIndex = ArrayPacking.unPackShorts(state.getLeftIndex(), state.isCompressed());
        short[] rightIndex = ArrayPacking.unPackShorts(state.getRightIndex(), state.isCompressed());
        int[] cutDimension = ArrayPacking.unPackInts(state.getCutDimension(), state.isCompressed());
        int freeIndexPointer = state.getFreeIndexPointer();
        int[] freeIndexes = ArrayPacking.unPackInts(state.getFreeIndexes(), state.isCompressed());
        double[] cutValue = Arrays.copyOf(state.getCutValueDouble(), state.getCutValueDouble().length);

        return new SmallNodeStore(capacity, leftIndex, rightIndex, cutDimension, cutValue, freeIndexes,
                freeIndexPointer);
    }

    @Override
    public NodeStoreState toState(SmallNodeStore model) {
        NodeStoreState state = new NodeStoreState();

        state.setCompressed(true);
        state.setCapacity(model.getCapacity());

        state.setLeftIndex(ArrayPacking.pack(model.leftIndex, state.isCompressed()));
        state.setRightIndex(ArrayPacking.pack(model.rightIndex, state.isCompressed()));
        state.setCutDimension(ArrayPacking.pack(model.cutDimension, state.isCompressed()));
        state.setFreeIndexes(ArrayPacking.pack(model.getFreeIndexes(), state.isCompressed()));
        state.setFreeIndexPointer(model.getFreeIndexPointer());
        state.setCutValueDouble(Arrays.copyOf(model.cutValue, model.cutValue.length));

        return state;
    }

}
