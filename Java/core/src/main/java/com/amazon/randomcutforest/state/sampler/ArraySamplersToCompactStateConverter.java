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

package com.amazon.randomcutforest.state.sampler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.amazon.randomcutforest.executor.Sequential;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreDoubleState;
import com.amazon.randomcutforest.store.PointStoreDouble;

/**
 * This class converts a SimpleStreamSampler to the compact form; maintaining a
 * pointstore.
 */
public class ArraySamplersToCompactStateConverter {

    private final Map<double[], Integer> pointMap;
    private final PointStoreDouble pointStoreDouble;
    private final List<CompactSamplerState> compactSamplerStates;
    boolean storeSequences;

    public ArraySamplersToCompactStateConverter(boolean storeSequences, int dimensions, int capacity) {
        pointMap = new HashMap<>();
        pointStoreDouble = new PointStoreDouble(dimensions, capacity);
        compactSamplerStates = new ArrayList<>();
        this.storeSequences = storeSequences;
    }

    public PointStoreDoubleState getPointStoreDoubleState() {
        return new PointStoreDoubleMapper().toState(pointStoreDouble);
    }

    public List<CompactSamplerState> getCompactSamplerStates() {
        return compactSamplerStates;
    }

    public void addSampler(SimpleStreamSampler<double[]> sampler) {
        CompactSamplerState compactState = new CompactSamplerState(sampler.size(), sampler.getCapacity(),
                storeSequences);

        int i = 0;
        for (Sequential<double[]> sample : sampler.getSequentialSamples()) {
            double[] ref = sample.getValue();

            if (pointMap.containsKey(ref)) {
                compactState.referenceArray[i] = pointMap.get(ref);
                pointStoreDouble.incrementRefCount(compactState.referenceArray[i]);
            } else {
                compactState.referenceArray[i] = pointStoreDouble.add(ref);
                pointMap.put(ref, compactState.referenceArray[i]);
            }
            compactState.weightArray[i] = sample.getWeight();
            if (storeSequences) {
                compactState.sequenceArray[i] = sample.getSequenceIndex();
            }

            i++;
        }

        compactSamplerStates.add(compactState);
    }

    public void addSamples(ArrayList<List<Sequential<double[]>>> sampleList, int capacity) {

        for (List<Sequential<double[]>> indivList : sampleList) {
            CompactSamplerState compactData = new CompactSamplerState(indivList.size(), capacity, storeSequences);
            for (int i = 0; i < indivList.size(); i++) {
                double[] ref = indivList.get(i).getValue();

                if (pointMap.containsKey(ref)) {
                    compactData.referenceArray[i] = pointMap.get(ref);
                    pointStoreDouble.incrementRefCount(compactData.referenceArray[i]);
                } else {
                    compactData.referenceArray[i] = pointStoreDouble.add(ref);
                    pointMap.put(ref, compactData.referenceArray[i]);
                }
                compactData.weightArray[i] = indivList.get(i).getWeight();
                if (storeSequences) {
                    compactData.sequenceArray[i] = indivList.get(i).getSequenceIndex();
                }
            }
            compactSamplerStates.add(compactData);
        }
    }
}
