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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;

@Getter
@Setter
public class PointStoreDoubleMapper implements IStateMapper<PointStoreDouble, PointStoreState> {
    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public PointStoreDouble toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getDoubleData(), "doubleData must not be null");
        checkArgument(!state.isSinglePrecisionSet(), "incorrect use");
        int capacity = state.getCapacity();
        int dimensions = state.getDimensions();
        short[] refCount = Arrays.copyOf(state.getRefCount(), capacity);
        double[] store = Arrays.copyOf(state.getDoubleData(), state.getCurrentCapacity() * dimensions);
        int freeIndexPointer = state.getFreeIndexPointer();
        int[] freeIndexes = new int[capacity];
        System.arraycopy(state.getFreeIndexes(), 0, freeIndexes, 0, freeIndexPointer + 1);
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] locationList = null;
        if (!state.isDirectMapLocation()) {
            locationList = new int[capacity];
            System.arraycopy(state.getLocationList(), 0, locationList, 0, state.getLocationList().length);
        }

        PointStore.Builder builder = new PointStore.Builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled())
                .dynamicResizingEnabled(state.isDynamicResizingEnabled()).currentCapacity(state.getCurrentCapacity())
                .capacity(capacity).shingleSize(state.getShingleSize()).dimensions(state.getDimensions())
                .locationList(locationList).nextTimeStamp(state.getLastTimeStamp())
                .startOfFreeSegment(startOfFreeSegment).refCount(refCount).freeIndexes(freeIndexes)
                .freeIndexPointer(freeIndexPointer).knownShingle(state.getInternalShingle());
        return new PointStoreDouble(builder, store);
    }

    @Override
    public PointStoreState toState(PointStoreDouble model) {
        if (!model.isDirectLocationMap()) {
            model.compact();
        }
        PointStoreState state = new PointStoreState();
        state.setDimensions(model.getDimensions());
        state.setCapacity(model.getCapacity());
        state.setShingleSize(model.getShingleSize());
        state.setDirectMapLocation(model.isDirectLocationMap());
        state.setInternalShingle(model.getInternalShingle());
        state.setLastTimeStamp(model.getNextTimeStamp());
        state.setDynamicResizingEnabled(model.isDynamicResizingEnabled());
        state.setInternalShinglingEnabled(model.isInternalShinglingEnabled());
        state.setRotationEnabled(model.isInternalRotationEnabled());
        state.setCurrentCapacity(model.getCurrentStoreCapacity());
        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setFreeIndexPointer(model.getFreeIndexPointer());
        state.setSinglePrecisionSet(false);
        int prefix = model.getValidPrefix();
        state.setDoubleData(Arrays.copyOf(model.getStore(), prefix * model.getDimensions()));
        state.setRefCount(Arrays.copyOf(model.getRefCount(), prefix));
        if (model.isDirectLocationMap()) {
            state.setDoubleData(Arrays.copyOf(model.getStore(), prefix * model.getDimensions()));
        } else {
            state.setLocationList(Arrays.copyOf(model.getLocationList(), prefix));
            // the below assumes that compact() is invoked as the first step
            state.setDoubleData(Arrays.copyOf(model.getStore(), model.getStartOfFreeSegment()));
        }
        state.setFreeIndexes(Arrays.copyOf(model.getFreeIndexes(), model.getFreeIndexPointer() + 1));
        return state;
    }

}
