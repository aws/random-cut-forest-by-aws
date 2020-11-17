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

import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.IContextualStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;

public class CompactSamplerMapper
        implements IContextualStateMapper<CompactSampler, CompactSamplerState, RandomCutForestState> {

    @Override
    public CompactSampler toModel(CompactSamplerState state, RandomCutForestState forestState, long seed) {
        CompactSampler sampler = new CompactSampler(forestState.getSampleSize(), forestState.getLambda(), seed,
                forestState.isStoreSequenceIndexesEnabled());
        sampler.reInitialize(state);
        return sampler;
    }

    @Override
    public CompactSamplerState toState(CompactSampler model) {
        CompactSamplerState state = new CompactSamplerState();
        state.setSize(model.size());
        state.setCapacity(model.getCapacity());
        state.setWeightArray(Arrays.copyOf(model.getWeightArray(), model.size()));
        state.setReferenceArray(Arrays.copyOf(model.getReferenceArray(), model.size()));
        if (model.isStoreSequenceIndexesEnabled()) {
            state.setSequenceArray(Arrays.copyOf(model.getSequenceArray(), model.size()));
        }
        return state;
    }
}
