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

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.Version;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class PointStoreDoubleMapper implements IStateMapper<PointStoreDouble, PointStoreState> {
    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    @Override
    public PointStoreDouble toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getPointData(), "pointData must not be null");
        checkArgument(Precision.valueOf(state.getPrecision()) == Precision.FLOAT_64,
                "precision must be " + Precision.FLOAT_64);
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        double[] store = ArrayPacking.unpackDoubles(state.getPointData(), state.getCurrentStoreCapacity() * dimensions);
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] refCount = ArrayPacking.unpackInts(state.getRefCount(), indexCapacity, state.isCompressed());
        int[] locationList = new int[indexCapacity];
        Arrays.fill(locationList, PointStoreDouble.INFEASIBLE_LOCN);
        int[] tempList = ArrayPacking.unpackInts(state.getLocationList(), state.isCompressed());
        System.arraycopy(tempList, 0, locationList, 0, tempList.length);
        if (!state.getVersion().equals(Version.V3_0)) {
            transformArray(locationList, dimensions / state.getShingleSize());
        }

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
        state.setVersion(Version.V3_0);
        state.setCompressed(compressionEnabled);
        state.setDimensions(model.getDimensions());
        state.setCapacity(model.getCapacity());
        state.setShingleSize(model.getShingleSize());
        state.setDirectLocationMap(false);
        state.setInternalShinglingEnabled(model.isInternalShinglingEnabled());
        state.setLastTimeStamp(model.getNextSequenceIndex());
        if (model.isInternalShinglingEnabled()) {
            state.setInternalShingle(model.getInternalShingle());
            state.setRotationEnabled(model.isInternalRotationEnabled());
        }
        state.setDynamicResizingEnabled(true);
        if (state.isDynamicResizingEnabled()) {
            state.setCurrentStoreCapacity(model.getCurrentStoreCapacity());
            state.setIndexCapacity(model.getIndexCapacity());
        }
        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setPrecision(Precision.FLOAT_64.name());
        // int prefix = model.getValidPrefix();
        int[] refcount = model.getRefCount();
        state.setRefCount(ArrayPacking.pack(refcount, refcount.length, state.isCompressed()));
        int[] locationList = model.getLocationList();
        state.setLocationList(ArrayPacking.pack(locationList, locationList.length, state.isCompressed()));
        state.setPointData(ArrayPacking.pack(model.getStore(), model.getStartOfFreeSegment()));
        return state;
    }

    void transformArray(int[] location, int baseDimension) {
        checkArgument(baseDimension > 0, "incorrect invocation");
        for (int i = 0; i < location.length; i++) {
            if (location[i] > 0) {
                location[i] = location[i] / baseDimension;
            }
        }
    }
}
