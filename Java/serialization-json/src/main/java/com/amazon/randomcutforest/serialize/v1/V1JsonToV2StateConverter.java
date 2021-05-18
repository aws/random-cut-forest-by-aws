/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.serialize.v1;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.ExecutionContext;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.fasterxml.jackson.databind.ObjectMapper;

public class V1JsonToV2StateConverter {

    private final ObjectMapper mapper = new ObjectMapper();

    public RandomCutForestState convert(String json) throws IOException {
        V1SerializedRandomCutForest forest = mapper.readValue(json, V1SerializedRandomCutForest.class);

        RandomCutForestState state = new RandomCutForestState();
        state.setNumberOfTrees(forest.getNumberOfTrees());
        state.setDimensions(forest.getDimensions());
        state.setTimeDecay(forest.getLambda());
        state.setSampleSize(forest.getSampleSize());
        state.setShingleSize(1);
        state.setCenterOfMassEnabled(forest.isCenterOfMassEnabled());
        state.setOutputAfter(forest.getOutputAfter());
        state.setStoreSequenceIndexesEnabled(forest.isStoreSequenceIndexesEnabled());
        state.setTotalUpdates(forest.getExecutor().getExecutor().getTotalUpdates());
        state.setCompact(true);
        state.setInternalShinglingEnabled(false);
        state.setBoundingBoxCacheFraction(1.0);
        state.setSaveSamplerStateEnabled(true);
        state.setSaveTreeStateEnabled(false);
        state.setSaveCoordinatorStateEnabled(true);
        state.setPrecision(Precision.FLOAT_64.name());
        state.setCompressed(false);
        state.setPartialTreeState(false);

        ExecutionContext executionContext = new ExecutionContext();
        executionContext.setParallelExecutionEnabled(forest.isParallelExecutionEnabled());
        executionContext.setThreadPoolSize(forest.getThreadPoolSize());
        state.setExecutionContext(executionContext);

        SamplerConverter samplerConverter = new SamplerConverter(state.getDimensions(),
                state.getNumberOfTrees() * state.getSampleSize());

        Arrays.stream(forest.getExecutor().getExecutor().getTreeUpdaters())
                .map(V1SerializedRandomCutForest.TreeUpdater::getSampler).forEach(samplerConverter::addSampler);

        state.setPointStoreState(samplerConverter.getPointStoreState());
        state.setCompactSamplerStates(samplerConverter.compactSamplerStates);

        return state;
    }

    static class SamplerConverter {
        private final Map<double[], Integer> pointMap;
        private final PointStoreDouble pointStore;
        private final List<CompactSamplerState> compactSamplerStates;

        public SamplerConverter(int dimensions, int capacity) {
            pointMap = new HashMap<>();
            pointStore = new PointStoreDouble(dimensions, capacity);
            compactSamplerStates = new ArrayList<>();
        }

        public PointStoreState getPointStoreState() {
            return new PointStoreDoubleMapper().toState(pointStore);
        }

        public void addSampler(V1SerializedRandomCutForest.Sampler sampler) {
            V1SerializedRandomCutForest.WeightedSamples[] samples = sampler.getWeightedSamples();
            int[] pointIndex = new int[samples.length];
            float[] weight = new float[samples.length];
            long[] sequenceIndex = new long[samples.length];

            for (int i = 0; i < samples.length; i++) {
                V1SerializedRandomCutForest.WeightedSamples sample = samples[i];
                double[] point = sample.getPoint();
                if (pointMap.containsKey(point)) {
                    pointIndex[i] = pointMap.get(point);
                    pointStore.incrementRefCount(pointIndex[i]);
                } else {
                    pointIndex[i] = pointStore.add(point, i);
                    pointMap.put(point, pointIndex[i]);
                }
                weight[i] = (float) sample.getWeight();
                sequenceIndex[i] = sample.getSequenceIndex();
            }

            CompactSamplerState samplerState = new CompactSamplerState();
            samplerState.setSize(samples.length);
            samplerState.setCapacity(sampler.getSampleSize());
            samplerState.setLambda(sampler.getLambda());
            samplerState.setPointIndex(pointIndex);
            samplerState.setWeight(weight);
            samplerState.setSequenceIndex(sequenceIndex);
            samplerState.setSequenceIndexOfMostRecentLambdaUpdate(0L);
            samplerState.setMaxSequenceIndex(sampler.getEntriesSeen());
            samplerState.setInitialAcceptFraction(1.0);

            compactSamplerStates.add(samplerState);
        }
    }
}
