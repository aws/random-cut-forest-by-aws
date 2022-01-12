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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.io.IOException;
import java.io.Reader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.ExecutionContext;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreFloatMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;
import com.amazon.randomcutforest.tree.ITree;
import com.fasterxml.jackson.databind.ObjectMapper;

public class V1JsonToV2StateConverter {

    private final ObjectMapper mapper = new ObjectMapper();

    public RandomCutForestState convert(String json, Precision precision) throws IOException {
        V1SerializedRandomCutForest forest = mapper.readValue(json, V1SerializedRandomCutForest.class);
        return convert(forest, precision);
    }

    public Optional<RandomCutForestState> convert(ArrayList<String> jsons, int numberOfTrees, Precision precision)
            throws IOException {
        ArrayList<V1SerializedRandomCutForest> forests = new ArrayList<>(jsons.size());
        int sum = 0;

        for (int i = 0; i < jsons.size(); i++) {
            V1SerializedRandomCutForest forest = mapper.readValue(jsons.get(i), V1SerializedRandomCutForest.class);
            forests.add(forest);
            sum += forest.getNumberOfTrees();
        }
        if (sum < numberOfTrees) {
            return Optional.empty();
        }
        return Optional.ofNullable(convert(forests, numberOfTrees, precision));
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
        return convert(Collections.singletonList(serializedForest), serializedForest.getNumberOfTrees(), precision);
    }

    static class SamplerConverter {
        private final IPointStore pointStore;
        private final List<CompactSamplerState> compactSamplerStates;
        private final Precision precision;
        private final ITree globalTree;
        private final int maxNumberOfTrees;

        public SamplerConverter(int dimensions, int capacity, Precision precision, int maxNumberOfTrees) {
            if (precision == Precision.FLOAT_64) {
                pointStore = new PointStoreDouble(dimensions, capacity);
                globalTree = new CompactRandomCutTreeDouble.Builder().pointStore(pointStore)
                        .maxSize(pointStore.getCapacity() + 1).storeSequenceIndexesEnabled(false)
                        .centerOfMassEnabled(false).boundingBoxCacheFraction(1.0).build();
            } else {
                pointStore = PointStore.builder().dimensions(dimensions).capacity(capacity).shingleSize(1)
                        .initialSize(capacity).build();
                globalTree = new CompactRandomCutTreeFloat.Builder().pointStore(pointStore)
                        .maxSize(pointStore.getCapacity() + 1).storeSequenceIndexesEnabled(false)
                        .centerOfMassEnabled(false).boundingBoxCacheFraction(1.0).build();
            }
            compactSamplerStates = new ArrayList<>();
            this.maxNumberOfTrees = maxNumberOfTrees;
            this.precision = precision;
        }

        public PointStoreState getPointStoreState(Precision precision) {
            if (precision == Precision.FLOAT_64) {
                return new PointStoreDoubleMapper().toState((PointStoreDouble) pointStore);
            } else {
                return new PointStoreFloatMapper().toState((PointStore) pointStore);
            }
        }

        public void addSampler(V1SerializedRandomCutForest.Sampler sampler) {
            if (compactSamplerStates.size() < maxNumberOfTrees) {
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

    /**
     * the function merges a collection of RCF-1.0 models with same model parameters
     * and fixes the number of trees in the new model (which has to be less or equal
     * than the sum of the old models) The conversion uses the execution context of
     * the first forest and can be adjusted subsequently by setters
     * 
     * @param serializedForests A non-empty list of forests (together having more
     *                          trees than numberOfTrees)
     * @param numberOfTrees     the new number of trees
     * @param precision         the precision of the new forest
     * @return a merged RCF with the first numberOfTrees trees
     */
    public RandomCutForestState convert(List<V1SerializedRandomCutForest> serializedForests, int numberOfTrees,
            Precision precision) {
        checkArgument(serializedForests.size() > 0, "incorrect usage of convert");
        checkArgument(numberOfTrees > 0, "incorrect parameter");
        int sum = 0;
        for (int i = 0; i < serializedForests.size(); i++) {
            sum += serializedForests.get(i).getNumberOfTrees();
        }
        checkArgument(sum >= numberOfTrees, "incorrect parameters");

        RandomCutForestState state = new RandomCutForestState();
        state.setNumberOfTrees(numberOfTrees);
        state.setDimensions(serializedForests.get(0).getDimensions());
        state.setTimeDecay(serializedForests.get(0).getLambda());
        state.setSampleSize(serializedForests.get(0).getSampleSize());
        state.setShingleSize(1);
        state.setCenterOfMassEnabled(serializedForests.get(0).isCenterOfMassEnabled());
        state.setOutputAfter(serializedForests.get(0).getOutputAfter());
        state.setStoreSequenceIndexesEnabled(serializedForests.get(0).isStoreSequenceIndexesEnabled());
        state.setTotalUpdates(serializedForests.get(0).getExecutor().getExecutor().getTotalUpdates());
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
        executionContext.setParallelExecutionEnabled(serializedForests.get(0).isParallelExecutionEnabled());
        executionContext.setThreadPoolSize(serializedForests.get(0).getThreadPoolSize());
        state.setExecutionContext(executionContext);

        SamplerConverter samplerConverter = new SamplerConverter(state.getDimensions(),
                state.getNumberOfTrees() * state.getSampleSize() + 1, precision, numberOfTrees);

        serializedForests.stream().flatMap(f -> Arrays.stream(f.getExecutor().getExecutor().getTreeUpdaters()))
                .limit(numberOfTrees).map(V1SerializedRandomCutForest.TreeUpdater::getSampler)
                .forEach(samplerConverter::addSampler);

        state.setPointStoreState(samplerConverter.getPointStoreState(precision));
        state.setCompactSamplerStates(samplerConverter.compactSamplerStates);

        return state;
    }
}
