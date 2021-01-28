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
import com.amazon.randomcutforest.store.LeafStore;

@Getter
@Setter
public class LeafStoreMapper implements IStateMapper<LeafStore, LeafStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public LeafStore toModel(LeafStoreState state, long seed) {
        int capacity = state.getPointIndex().length;
        int[] pointIndex = Arrays.copyOf(state.getPointIndex(), capacity);
        short[] parentIndex = Arrays.copyOf(state.getParentIndex(), capacity);
        short[] mass = Arrays.copyOf(state.getMass(), capacity);

        short freeIndexPointer = (short) (state.getFreeIndexes().length - 1);
        short[] freeIndexes = new short[capacity];
        System.arraycopy(state.getFreeIndexes(), 0, freeIndexes, 0, state.getFreeIndexes().length);

        return new LeafStore(pointIndex, parentIndex, mass, freeIndexes, freeIndexPointer);
    }

    @Override
    public LeafStoreState toState(LeafStore model) {
        LeafStoreState state = new LeafStoreState();
        state.setPointIndex(Arrays.copyOf(model.pointIndex, model.pointIndex.length));
        state.setParentIndex(Arrays.copyOf(model.parentIndex, model.parentIndex.length));
        state.setMass(Arrays.copyOf(model.mass, model.mass.length));
        state.setFreeIndexes(Arrays.copyOf(model.getFreeIndexes(), model.getFreeIndexPointer() + 1));
        return state;
    }

}
