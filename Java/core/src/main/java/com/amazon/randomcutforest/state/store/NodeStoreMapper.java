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
import com.amazon.randomcutforest.store.NodeStore;

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
        int capacity = state.getLeftIndex().length;
        short[] leftIndex = Arrays.copyOf(state.getLeftIndex(), capacity);
        short[] rightIndex = Arrays.copyOf(state.getRightIndex(), capacity);
        short[] parentIndex = Arrays.copyOf(state.getParentIndex(), capacity);
        short[] mass = Arrays.copyOf(state.getMass(), capacity);
        int[] cutDimension = Arrays.copyOf(state.getCutDimension(), capacity);
        double[] cutValue = Arrays.copyOf(state.getCutValue(), capacity);

        short freeIndexPointer = (short) (state.getFreeIndexes().length - 1);
        short[] freeIndexes = new short[capacity];
        System.arraycopy(state.getFreeIndexes(), 0, freeIndexes, 0, freeIndexPointer + 1);

        return new NodeStore(parentIndex, leftIndex, rightIndex, cutDimension, cutValue, mass, freeIndexes,
                freeIndexPointer);
    }

    @Override
    public NodeStoreState toState(NodeStore model) {
        NodeStoreState state = new NodeStoreState();
        state.setLeftIndex(Arrays.copyOf(model.leftIndex, model.leftIndex.length));
        state.setRightIndex(Arrays.copyOf(model.rightIndex, model.rightIndex.length));
        state.setParentIndex(Arrays.copyOf(model.parentIndex, model.parentIndex.length));
        state.setMass(Arrays.copyOf(model.mass, model.mass.length));
        state.setCutDimension(Arrays.copyOf(model.cutDimension, model.cutDimension.length));
        state.setCutValue(Arrays.copyOf(model.cutValue, model.cutValue.length));
        state.setFreeIndexes(Arrays.copyOf(model.getFreeIndexes(), model.getFreeIndexPointer() + 1));
        return state;
    }
}
