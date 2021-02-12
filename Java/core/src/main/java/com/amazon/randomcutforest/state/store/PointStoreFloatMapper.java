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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.PointStoreFloat;

@Getter
@Setter
public class PointStoreFloatMapper implements IStateMapper<PointStoreFloat, PointStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public PointStoreFloat toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getFloatData(), "floatdata must not be null");

        int capacity = state.getCapacity();
        int dimensions = state.getDimensions();
        short[] refCount = Arrays.copyOf(state.getRefCount(), capacity);
        float[] store = Arrays.copyOf(state.getFloatData(), capacity * dimensions);
        int freeIndexPointer = state.getFreeIndexes().length - 1;
        int[] freeIndexes = new int[capacity];
        System.arraycopy(state.getFreeIndexes(), 0, freeIndexes, 0, freeIndexPointer + 1);

        return new PointStoreFloat(store, refCount, freeIndexes, freeIndexPointer);
    }

    @Override
    public PointStoreState toState(PointStoreFloat model) {
        PointStoreState state = new PointStoreState();
        state.setCapacity(model.getCapacity());
        state.setDimensions(model.getDimensions());
        int prefix = model.getValidPrefix();
        state.setFloatData(Arrays.copyOf(model.getStore(), prefix * model.getDimensions()));
        state.setRefCount(Arrays.copyOf(model.getRefCount(), prefix));
        state.setFreeIndexes(Arrays.copyOf(model.getFreeIndexes(), model.getFreeIndexPointer() + 1));
        return state;
    }
}
