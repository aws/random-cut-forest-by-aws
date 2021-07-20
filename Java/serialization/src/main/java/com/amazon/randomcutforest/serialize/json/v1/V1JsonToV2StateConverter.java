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

package com.amazon.randomcutforest.serialize.json.v1;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.ExecutionContext;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreFloatMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;
import com.amazon.randomcutforest.tree.ITree;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.Reader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class V1JsonToV2StateConverter {

    private final ObjectMapper mapper = new ObjectMapper();

    public RandomCutForestState convert(String json, Precision precision) throws IOException {
        V1SerializedRandomCutForest forest = mapper.readValue(json, V1SerializedRandomCutForest.class);
        return convert(forest, precision);
    }

    public RandomCutForestState convert(Reader reader, Precision precision) throws IOException {
        V1SerializedRandomCutForest forest = mapper.readValue(reader, V1SerializedRandomCutForest.class);
        return convert(forest, precision);
    }

    public RandomCutForestState convert(URL url, Precision precision) throws IOException {
        V1SerializedRandomCutForest forest = mapper.readValue(url, V1SerializedRandomCutForest.class);
        return convert(forest, precision);
    }

    public RandomCutForestState convert(V1SerializedRandomCutForest serializedForest, Precision precision) {
        RandomCutForestState state = new RandomCutForestState();
        state.setNumberOfTrees(serializedForest.getNumberOfTrees());
        state.setDimensions(serializedForest.getDimensions());
        state.setTimeDecay(serializedForest.getLambda());
        state.setSampleSize(serializedForest.getSampleSize());
        state.setShingleSize(1);
        state.setCenterOfMassEnabled(serializedForest.isCenterOfMassEnabled());
        state.setOutputAfter(serializedForest.getOutputAfter());
        state.setStoreSequenceIndexesEnabled(serializedForest.isStoreSequenceIndexesEnabled());
        state.setTotalUpdates(serializedForest.getExecutor().getExecutor().getTotalUpdates());
        state.setCompact(true);
        state.setInternalShinglingEnabled(false);
        state.setBoundingBoxCacheFraction(1.0);
        state.setSaveSamplerStateEnabled(true);
        state.setSaveTreeStateEnabled(false);
        state.setSaveCoordinatorStateEnabled(true);
        state.setPrecision(precision.name());
        state.setCompressed(false);
        state.setPartialTreeState(false);

        ExecutionContext executionContext = new ExecutionContext();
        executionContext.setParallelExecutionEnabled(serializedForest.isParallelExecutionEnabled());
        executionContext.setThreadPoolSize(serializedForest.getThreadPoolSize());
        state.setExecutionContext(executionContext);

        SamplerConverter samplerConverter = new SamplerConverter(state.getDimensions(),
                state.getNumberOfTrees() * state.getSampleSize() + 1, precision);

        Arrays.stream(serializedForest.getExecutor().getExecutor().getTreeUpdaters())
                .map(V1SerializedRandomCutForest.TreeUpdater::getSampler).forEach(samplerConverter::addSampler);

        state.setPointStoreState(samplerConverter.getPointStoreState(precision));
        state.setCompactSamplerStates(samplerConverter.compactSamplerStates);

        return state;
    }

    static class SamplerConverter {
        private final IPointStore pointStore;
        private final List<CompactSamplerState> compactSamplerStates;
        private final Precision precision;
        private final ITree globalTree;

        public SamplerConverter(int dimensions, int capacity, Precision precision) {
            if (precision == Precision.FLOAT_64) {
                pointStore = new PointStoreDouble(dimensions, capacity);
                globalTree = new CompactRandomCutTreeDouble.Builder().pointStore(pointStore)
                        .maxSize(pointStore.getCapacity() + 1).storeSequenceIndexesEnabled(false)
                        .centerOfMassEnabled(false).boundingBoxCacheFraction(1.0).build();
            } else {
                pointStore = new PointStoreFloat(dimensions, capacity);
                globalTree = new CompactRandomCutTreeFloat.Builder().pointStore(pointStore)
                        .maxSize(pointStore.getCapacity() + 1).storeSequenceIndexesEnabled(false)
                        .centerOfMassEnabled(false).boundingBoxCacheFraction(1.0).build();
            }
            compactSamplerStates = new ArrayList<>();
            this.precision = precision;
        }

        public PointStoreState getPointStoreState(Precision precision) {
            if (precision == Precision.FLOAT_64) {
                return new PointStoreDoubleMapper().toState((PointStoreDouble) pointStore);
            } else {
                return new PointStoreFloatMapper().toState((PointStoreFloat) pointStore);
            }
        }

        public void addSampler(V1SerializedRandomCutForest.Sampler sampler) {
            V1SerializedRandomCutForest.WeightedSamples[] samples = sampler.getWeightedSamples();
            int[] pointIndex = new int[samples.length];
            float[] weight = new float[samples.length];
            long[] sequenceIndex = new long[samples.length];

            for (int i = 0; i < samples.length; i++) {
                V1SerializedRandomCutForest.WeightedSamples sample = samples[i];
                double[] point = sample.getPoint();
                int index = pointStore.add(point, sample.getSequenceIndex());
                pointIndex[i] = (Integer) globalTree.addPoint(index, 0L);
                if (pointIndex[i] != index) {
                    pointStore.incrementRefCount(pointIndex[i]);
                    pointStore.decrementRefCount(index);
                }
                weight[i] = (float) sample.getWeight();
                sequenceIndex[i] = sample.getSequenceIndex();
            }

            CompactSamplerState samplerState = new CompactSamplerState();
            samplerState.setSize(samples.length);
            samplerState.setCapacity(sampler.getSampleSize());
            samplerState.setTimeDecay(sampler.getLambda());
            samplerState.setPointIndex(pointIndex);
            samplerState.setWeight(weight);
            samplerState.setSequenceIndex(sequenceIndex);
            samplerState.setSequenceIndexOfMostRecentTimeDecayUpdate(0L);
            samplerState.setMaxSequenceIndex(sampler.getEntriesSeen());
            samplerState.setInitialAcceptFraction(1.0);

            compactSamplerStates.add(samplerState);
        }
    }
}
