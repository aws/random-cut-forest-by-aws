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

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class CompactSamplerMapper implements IStateMapper<CompactSampler, CompactSamplerState> {

    /**
     * This flag is passed to the constructor for {@code CompactSampler} when a new
     * sampler is constructed in {@link #toModel}. If true, then the sampler will
     * validate that the weight array in a {@code CompactSamplerState} instance
     * satisfies the heap property. The heap property is not validated by default.
     */
    private boolean validateHeapEnabled = false;

    /**
     * used to compress data, can be set to false for debug
     */
    private boolean compressionEnabled = true;

    @Override
    public CompactSampler toModel(CompactSamplerState state, long seed) {
        float[] weight = new float[state.getCapacity()];
        int[] pointIndex = new int[state.getCapacity()];
        long[] sequenceIndex;

        int size = state.getSize();
        System.arraycopy(state.getWeight(), 0, weight, 0, size);
        System.arraycopy(ArrayPacking.unpackInts(state.getPointIndex(), state.isCompressed()), 0, pointIndex, 0, size);
        if (state.isStoreSequenceIndicesEnabled()) {
            sequenceIndex = new long[state.getCapacity()];
            System.arraycopy(state.getSequenceIndex(), 0, sequenceIndex, 0, size);
        } else {
            sequenceIndex = null;
        }

        return new CompactSampler.Builder<>().capacity(state.getCapacity()).timeDecay(state.getTimeDecay())
                .randomSeed(state.getRandomSeed()).storeSequenceIndexesEnabled(state.isStoreSequenceIndicesEnabled())
                .weight(weight).pointIndex(pointIndex).sequenceIndex(sequenceIndex).validateHeap(validateHeapEnabled)
                .initialAcceptFraction(state.getInitialAcceptFraction())
                .mostRecentTimeDecayUpdate(state.getSequenceIndexOfMostRecentTimeDecayUpdate())
                .maxSequenceIndex(state.getMaxSequenceIndex()).size(state.getSize()).build();
    }

    @Override
    public CompactSamplerState toState(CompactSampler model) {
        CompactSamplerState state = new CompactSamplerState();
        state.setSize(model.size());
        state.setCompressed(compressionEnabled);
        state.setCapacity(model.getCapacity());
        state.setTimeDecay(model.getTimeDecay());
        state.setSequenceIndexOfMostRecentTimeDecayUpdate(model.getMostRecentTimeDecayUpdate());
        state.setMaxSequenceIndex(model.getMaxSequenceIndex());
        state.setInitialAcceptFraction(model.getInitialAcceptFraction());
        state.setStoreSequenceIndicesEnabled(model.isStoreSequenceIndexesEnabled());
        state.setRandomSeed(model.getRandomSeed());

        state.setWeight(Arrays.copyOf(model.getWeightArray(), model.size()));
        state.setPointIndex(ArrayPacking.pack(model.getPointIndexArray(), model.size(), state.isCompressed()));
        if (model.isStoreSequenceIndexesEnabled()) {
            state.setSequenceIndex(Arrays.copyOf(model.getSequenceIndexArray(), model.size()));

        }

        return state;
    }
}
