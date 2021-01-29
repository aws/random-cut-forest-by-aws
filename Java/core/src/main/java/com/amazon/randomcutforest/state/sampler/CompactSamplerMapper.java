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

import java.util.Arrays;
import java.util.Random;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class CompactSamplerMapper implements IStateMapper<CompactSampler, CompactSamplerState> {

    /**
     * This flag is passed to the constructor for {@code CompactSampler} when a new
     * sampler is constructed in {@link #toModel}. If true, then the sampler will
     * validate that the weight array in a {@code CompactSamplerState} instance
     * satisfies the heap property. The heap property is not validated by default.
     */
    private boolean validateHeap = false;

    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    @Override
    public CompactSampler toModel(CompactSamplerState state, long seed) {
        float[] weight;
        int[] pointIndex;
        long[] sequenceIndex;

        if (copy) {
            weight = new float[state.getCapacity()];
            pointIndex = new int[state.getCapacity()];

            int size = state.getSize();
            System.arraycopy(state.getWeight(), 0, weight, 0, size);
            System.arraycopy(state.getPointIndex(), 0, pointIndex, 0, size);
            if (state.getSequenceIndex() != null) {
                sequenceIndex = new long[state.getCapacity()];
                System.arraycopy(state.getSequenceIndex(), 0, sequenceIndex, 0, size);
            } else {
                sequenceIndex = null;
            }
        } else {
            weight = state.getWeight();
            pointIndex = state.getPointIndex();
            sequenceIndex = state.getSequenceIndex();
        }

        return new CompactSampler(state.getCapacity(), state.getSize(), state.getLambda(), new Random(seed), weight,
                pointIndex, sequenceIndex, validateHeap);
    }

    @Override
    public CompactSamplerState toState(CompactSampler model) {
        CompactSamplerState state = new CompactSamplerState();
        state.setSize(model.size());
        state.setCapacity(model.getCapacity());
        state.setLambda(model.getLambda());

        if (copy) {
            state.setWeight(Arrays.copyOf(model.getWeightArray(), model.size()));
            state.setPointIndex(Arrays.copyOf(model.getPointIndexArray(), model.size()));
            if (model.isStoreSequenceIndexesEnabled()) {
                state.setSequenceIndex(Arrays.copyOf(model.getSequenceIndexArray(), model.size()));
            }
        } else {
            state.setWeight(model.getWeightArray());
            state.setPointIndex(model.getPointIndexArray());
            if (model.isStoreSequenceIndexesEnabled()) {
                state.setSequenceIndex(model.getSequenceIndexArray());
            }
        }
        return state;
    }
}
