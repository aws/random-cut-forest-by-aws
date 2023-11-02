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

package com.amazon.randomcutforest.serialize.json.v2;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.state.Version.V2_0;
import static com.amazon.randomcutforest.state.Version.V2_1;

import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.state.Version;
import com.amazon.randomcutforest.state.sampler.CompactSamplerMapper;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeContext;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.ArrayPacking;

public class V2StateToV3ForestConverter {

    public RandomCutForest convert(RandomCutForestState v2State) {
        String version = v2State.getVersion();
        checkArgument(version.equals(V2_0) || version.equals(V2_1), "incorrect convertor");
        if (Precision.valueOf(v2State.getPrecision()) == Precision.FLOAT_32) {
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setCompressionEnabled(v2State.isCompressed());
            return mapper.toModel(v2State);
        } else {
            return convertFrom64(v2State);
        }
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

    RandomCutForest convertFrom64(RandomCutForestState state) {
        boolean parallel = false;
        int threadPoolSize = 1;

        if (state.getExecutionContext() != null) {
            parallel = state.getExecutionContext().isParallelExecutionEnabled();
            threadPoolSize = state.getExecutionContext().getThreadPoolSize();
        }
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().numberOfTrees(state.getNumberOfTrees())
                .dimensions(state.getDimensions()).timeDecay(state.getTimeDecay()).sampleSize(state.getSampleSize())
                .centerOfMassEnabled(state.isCenterOfMassEnabled()).outputAfter(state.getOutputAfter())
                .parallelExecutionEnabled(parallel).threadPoolSize(threadPoolSize)
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).shingleSize(state.getShingleSize())
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction()).compact(state.isCompact())
                .internalShinglingEnabled(state.isInternalShinglingEnabled());
        Random random = builder.getRandom();
        PointStore pointStore = convertFromDouble(state.getPointStoreState());
        ComponentList<Integer, float[]> components = new ComponentList<>();

        PointStoreCoordinator<float[]> coordinator = new PointStoreCoordinator<>(pointStore);
        coordinator.setTotalUpdates(state.getTotalUpdates());
        CompactRandomCutTreeContext context = new CompactRandomCutTreeContext();
        context.setPointStore(pointStore);
        context.setMaxSize(state.getSampleSize());
        checkArgument(state.isSaveSamplerStateEnabled(), " conversion cannot proceed without samplers");
        List<CompactSamplerState> samplerStates = state.getCompactSamplerStates();
        CompactSamplerMapper samplerMapper = new CompactSamplerMapper();

        for (int i = 0; i < state.getNumberOfTrees(); i++) {
            CompactSampler compactData = samplerMapper.toModel(samplerStates.get(i));
            RandomCutTree tree = RandomCutTree.builder().capacity(state.getSampleSize()).pointStoreView(pointStore)
                    .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled())
                    .outputAfter(state.getOutputAfter()).centerOfMassEnabled(state.isCenterOfMassEnabled())
                    .randomSeed(random.nextLong()).build();
            CompactSampler sampler = CompactSampler.builder().capacity(state.getSampleSize())
                    .timeDecay(state.getTimeDecay()).randomSeed(random.nextLong()).build();
            sampler.setMaxSequenceIndex(compactData.getMaxSequenceIndex());
            sampler.setMostRecentTimeDecayUpdate(compactData.getMostRecentTimeDecayUpdate());

            for (Weighted<Integer> sample : compactData.getWeightedSample()) {
                Integer reference = sample.getValue();
                Integer newReference = tree.addPoint(reference, sample.getSequenceIndex());
                if (newReference.intValue() != reference.intValue()) {
                    pointStore.incrementRefCount(newReference);
                    pointStore.decrementRefCount(reference);
                }
                sampler.addPoint(newReference, sample.getWeight(), sample.getSequenceIndex());
            }
            components.add(new SamplerPlusTree<>(sampler, tree));
        }

        RandomCutForest forest = new RandomCutForest(builder, coordinator, components, random);
        if (!state.isCurrentlySampling()) {
            forest.pauseSampling();
        }
        return forest;
    }
}
