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
import com.amazon.randomcutforest.util.ArrayPacking;

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
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        double[] store = Arrays.copyOf(state.getDoubleData(), state.getCurrentStoreCapacity() * dimensions);
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] refCount = Arrays.copyOf(ArrayPacking.unPackInts(state.getRefCount(), state.isCompressed()),
                indexCapacity);
        int[] locationList = new int[indexCapacity];
        Arrays.fill(locationList, PointStore.INFEASIBLE_POINTSTORE_LOCATION);
        int[] tempList = ArrayPacking.unPackInts(state.getLocationList(), state.isCompressed());
        System.arraycopy(tempList, 0, locationList, 0, tempList.length);

        return PointStoreDouble.builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled())
                .dynamicResizingEnabled(state.isDynamicResizingEnabled())
                .directLocationEnabled(state.isDirectLocationMap()).indexCapacity(indexCapacity)
                .currentStoreCapacity(state.getCurrentStoreCapacity()).capacity(state.getCapacity())
                .shingleSize(state.getShingleSize()).dimensions(state.getDimensions()).locationList(locationList)
                .nextTimeStamp(state.getLastTimeStamp()).startOfFreeSegment(startOfFreeSegment).refCount(refCount)
                .knownShingle(state.getInternalShingle()).store(store).build();
    }

    @Override
    public PointStoreState toState(PointStoreDouble model) {
        model.compact();
        PointStoreState state = new PointStoreState();
        state.setCompressed(true);
        state.setDimensions(model.getDimensions());
        state.setCapacity(model.getCapacity());
        state.setShingleSize(model.getShingleSize());
        state.setDirectLocationMap(model.isDirectLocationMap());
        state.setInternalShingle(model.getInternalShingle());
        state.setLastTimeStamp(model.getNextSequenceIndex());
        state.setDynamicResizingEnabled(model.isDynamicResizingEnabled());
        state.setInternalShinglingEnabled(model.isInternalShinglingEnabled());
        state.setRotationEnabled(model.isInternalRotationEnabled());
        state.setCurrentStoreCapacity(model.getCurrentStoreCapacity());
        state.setIndexCapacity(model.getIndexCapacity());
        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setSinglePrecisionSet(false);
        int prefix = model.getValidPrefix();
        state.setRefCount(ArrayPacking.pack(Arrays.copyOf(model.getRefCount(), prefix), state.isCompressed()));
        state.setLocationList(ArrayPacking.pack(Arrays.copyOf(model.getLocationList(), prefix), state.isCompressed()));
        state.setDoubleData(Arrays.copyOf(model.getStore(), model.getStartOfFreeSegment()));
        return state;
    }

}
