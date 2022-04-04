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
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.Version;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.store.PointStoreLarge;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class PointStoreMapper implements IStateMapper<PointStore, PointStoreState> {

    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    private int numberOfTrees = 255; // byte encoding as default

    @Override
    public PointStore toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getPointData(), "pointData must not be null");
        checkArgument(Precision.valueOf(state.getPrecision()) == Precision.FLOAT_32,
                "precision must be " + Precision.FLOAT_32);
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        float[] store = ArrayPacking.unpackFloats(state.getPointData(), state.getCurrentStoreCapacity() * dimensions);
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] refCount = ArrayPacking.unpackInts(state.getRefCount(), indexCapacity, state.isCompressed());
        int[] locationList = new int[indexCapacity];
        int[] tempList = ArrayPacking.unpackInts(state.getLocationList(), state.isCompressed());
        if (!state.getVersion().equals(Version.V3_0)) {
            int shingleSize = state.getShingleSize();
            int baseDimension = dimensions / shingleSize;
            for (int i = 0; i < tempList.length; i++) {
                checkArgument(tempList[i] % baseDimension == 0,
                        "the location field should be a multiple of dimension/shingle size for versions before 3.0");
                locationList[i] = tempList[i] / baseDimension;
            }
        } else {
            int[] duplicateRefs = null;
            if (state.getDuplicateRefs() != null) {
                duplicateRefs = ArrayPacking.unpackInts(state.getDuplicateRefs(), state.isCompressed());
                checkArgument(duplicateRefs.length % 2 == 0, " corrupt duplicates");
                for (int i = 0; i < duplicateRefs.length; i += 2) {
                    refCount[duplicateRefs[i]] += duplicateRefs[i + 1];
                }
            }
            int nextLocation = 0;
            for (int i = 0; i < indexCapacity; i++) {
                if (refCount[i] > 0) {
                    locationList[i] = tempList[nextLocation];
                    ++nextLocation;
                } else {
                    locationList[i] = PointStoreLarge.INFEASIBLE_LOCN;
                }
            }
            checkArgument(nextLocation == tempList.length, "incorrect location encoding");
        }

        return PointStore.builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled())
                .dynamicResizingEnabled(state.isDynamicResizingEnabled())
                .directLocationEnabled(state.isDirectLocationMap()).indexCapacity(indexCapacity)
                .currentStoreCapacity(state.getCurrentStoreCapacity()).capacity(state.getCapacity())
                .shingleSize(state.getShingleSize()).dimensions(state.getDimensions()).locationList(locationList)
                .nextTimeStamp(state.getLastTimeStamp()).startOfFreeSegment(startOfFreeSegment).refCount(refCount)
                .knownShingle(state.getInternalShingle()).store(store).build();
    }

    @Override
    public PointStoreState toState(PointStore model) {
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
            state.setInternalShingle(toDoubleArray(model.getInternalShingle()));
            state.setRotationEnabled(model.isInternalRotationEnabled());
        }
        state.setDynamicResizingEnabled(true);
        if (state.isDynamicResizingEnabled()) {
            state.setCurrentStoreCapacity(model.getCurrentStoreCapacity());
            state.setIndexCapacity(model.getIndexCapacity());
        }
        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setPrecision(Precision.FLOAT_32.name());
        int[] refcount = model.getRefCount();
        int[] tempList = model.getLocationList();
        int[] locationList = new int[model.getIndexCapacity()];
        int[] duplicateRefs = new int[2 * model.getIndexCapacity()];
        int size = 0;
        int duplicateSize = 0;
        for (int i = 0; i < refcount.length; i++) {
            if (refcount[i] > 0) {
                locationList[size] = tempList[i];
                ++size;
                if (refcount[i] > numberOfTrees) {
                    duplicateRefs[duplicateSize] = i;
                    duplicateRefs[duplicateSize + 1] = refcount[i] - numberOfTrees;
                    refcount[i] = numberOfTrees;
                    duplicateSize += 2;
                }
            }
        }
        state.setRefCount(ArrayPacking.pack(refcount, refcount.length, state.isCompressed()));
        state.setDuplicateRefs(ArrayPacking.pack(duplicateRefs, duplicateSize, state.isCompressed()));
        state.setLocationList(ArrayPacking.pack(locationList, size, state.isCompressed()));
        state.setPointData(ArrayPacking.pack(model.getStore(), model.getStartOfFreeSegment()));
        return state;
    }

    public PointStore convertFromDouble(PointStoreState state) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getPointData(), "pointData must not be null");
        checkArgument(Precision.valueOf(state.getPrecision()) == Precision.FLOAT_64,
                "precision must be " + Precision.FLOAT_64);
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        float[] store = toFloatArray(
                ArrayPacking.unpackDoubles(state.getPointData(), state.getCurrentStoreCapacity() * dimensions));
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] refCount = ArrayPacking.unpackInts(state.getRefCount(), indexCapacity, state.isCompressed());
        int[] locationList = new int[indexCapacity];
        int[] tempList = ArrayPacking.unpackInts(state.getLocationList(), state.isCompressed());
        System.arraycopy(tempList, 0, locationList, 0, tempList.length);
        if (!state.getVersion().equals(Version.V3_0)) {
            transformArray(locationList, dimensions / state.getShingleSize());
        }

        return PointStore.builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled())
                .dynamicResizingEnabled(state.isDynamicResizingEnabled())
                .directLocationEnabled(state.isDirectLocationMap()).indexCapacity(indexCapacity)
                .currentStoreCapacity(state.getCurrentStoreCapacity()).capacity(state.getCapacity())
                .shingleSize(state.getShingleSize()).dimensions(state.getDimensions()).locationList(locationList)
                .nextTimeStamp(state.getLastTimeStamp()).startOfFreeSegment(startOfFreeSegment).refCount(refCount)
                .knownShingle(state.getInternalShingle()).store(store).build();
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
