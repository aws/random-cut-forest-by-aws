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

package com.amazon.randomcutforest.serialize.V1ToV2Converter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.store.PointStoreDouble;

/**
 * This class converts a SimpleStreamSampler to the compact form; maintaining a
 * pointstore.
 */
public class ArraySamplersToCompactStateConverter {

    private final Map<double[], Integer> pointMap;
    private final PointStoreDouble pointStoreDouble;
    private final List<CompactSamplerState> compactSamplerStates;
    boolean storeSequenceIndexesEnabled;

    public ArraySamplersToCompactStateConverter(boolean storeSequenceIndexesEnabled, int dimensions, int capacity) {
        pointMap = new HashMap<>();
        pointStoreDouble = new PointStoreDouble(dimensions, capacity);
        compactSamplerStates = new ArrayList<>();
        this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
    }

    public PointStoreState getPointStoreDoubleState() {
        return new PointStoreDoubleMapper().toState(pointStoreDouble);
    }

    public List<CompactSamplerState> getCompactSamplerStates() {
        return compactSamplerStates;
    }

    public void addSampler(SimpleStreamSampler<double[]> sampler) {
        int[] pointIndex = new int[sampler.size()];
        float[] weight = new float[sampler.size()];
        long[] sequenceIndex = storeSequenceIndexesEnabled ? new long[sampler.size()] : null;

        int i = 0;
        for (Weighted<double[]> sample : sampler.getWeightedSample()) {
            double[] index = sample.getValue();

            if (pointMap.containsKey(index)) {
                pointIndex[i] = pointMap.get(index);
                pointStoreDouble.incrementRefCount(pointIndex[i]);
            } else {
                pointIndex[i] = pointStoreDouble.add(index, i);
                pointMap.put(index, pointIndex[i]);
            }
            weight[i] = sample.getWeight();
            if (sequenceIndex != null) {
                sequenceIndex[i] = sample.getSequenceIndex();
            }

            i++;
        }

        CompactSamplerState samplerState = new CompactSamplerState();
        samplerState.setSize(sampler.size());
        samplerState.setCapacity(sampler.getCapacity());
        samplerState.setTimeDecay(sampler.getTimeDecay());
        samplerState.setPointIndex(pointIndex);
        samplerState.setWeight(weight);
        samplerState.setSequenceIndex(sequenceIndex);
        samplerState.setSequenceIndexOfMostRecentTimeDecayUpdate(sampler.getMostRecentTimeDecayUpdate());
        samplerState.setMaxSequenceIndex(sampler.getMaxSequenceIndex());
        samplerState.setInitialAcceptFraction(sampler.getInitialAcceptFraction());

        compactSamplerStates.add(samplerState);
    }
}
