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
import com.amazon.randomcutforest.store.SmallLeafStore;

@Getter
@Setter
public class SmallLeafStoreMapper implements IStateMapper<SmallLeafStore, LeafStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public SmallLeafStore toModel(LeafStoreState state, long seed) {
        int capacity = state.getCapacity();
        int[] pointIndex = Arrays.copyOf(state.getPointIndex(), capacity);
        short[] parentIndex = Arrays.copyOf(state.getSmallParentIndex(), capacity);
        short[] mass = Arrays.copyOf(state.getSmallMass(), capacity);
        short freeIndexPointer = (short) (state.getFreeIndexPointer());
        short[] freeIndexes = Arrays.copyOf(state.getSmallFreeIndexes(), state.getSmallFreeIndexes().length);

        return new SmallLeafStore(capacity, pointIndex, parentIndex, mass, freeIndexes, freeIndexPointer);
    }

    @Override
    public LeafStoreState toState(SmallLeafStore model) {
        LeafStoreState state = new LeafStoreState();
        state.setCapacity(model.getCapacity());
        state.setPointIndex(Arrays.copyOf(model.pointIndex, model.pointIndex.length));
        state.setSmallParentIndex(Arrays.copyOf(model.parentIndex, model.parentIndex.length));
        state.setSmallMass(Arrays.copyOf(model.mass, model.mass.length));
        state.setFreeIndexPointer(model.getFreeIndexPointer());
        state.setSmallFreeIndexes(Arrays.copyOf(model.getFreeIndexes(), model.getFreeIndexes().length));
        return state;
    }

}
