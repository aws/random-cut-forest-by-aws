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

package com.amazon.randomcutforest.parkservices.state.returntypes;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.RCFComputeDescriptor;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.returntypes.DiVectorMapper;

@Getter
@Setter
public class ComputeDescriptorMapper implements IStateMapper<RCFComputeDescriptor, ComputeDescriptorState> {

    @Override
    public RCFComputeDescriptor toModel(ComputeDescriptorState state, long seed) {

        RCFComputeDescriptor descriptor = new RCFComputeDescriptor(null, 0L);
        descriptor.setRCFScore(state.getLastAnomalyScore());
        descriptor.setInternalTimeStamp(state.getLastAnomalyTimeStamp());
        descriptor.setAttribution(new DiVectorMapper().toModel(state.getLastAnomalyAttribution()));
        descriptor.setRCFPoint(state.getLastAnomalyPoint());
        descriptor.setExpectedRCFPoint(state.getLastExpectedPoint());
        descriptor.setRelativeIndex(state.getLastRelativeIndex());
        descriptor.setScoringStrategy(ScoringStrategy.valueOf(state.getLastStrategy()));
        descriptor.setShift(state.getLastShift());
        descriptor.setPostShift(state.getLastPostShift());
        descriptor.setTransformDecay(state.getTransformDecay());
        return descriptor;
    }

    @Override
    public ComputeDescriptorState toState(RCFComputeDescriptor descriptor) {

        ComputeDescriptorState state = new ComputeDescriptorState();
        state.setLastAnomalyTimeStamp(descriptor.getInternalTimeStamp());
        state.setLastAnomalyScore(descriptor.getRCFScore());
        state.setLastAnomalyAttribution(new DiVectorMapper().toState(descriptor.getAttribution()));
        state.setLastAnomalyPoint(descriptor.getRCFPoint());
        state.setLastExpectedPoint(descriptor.getExpectedRCFPoint());
        state.setLastRelativeIndex(descriptor.getRelativeIndex());
        state.setLastStrategy(descriptor.getScoringStrategy().name());
        state.setLastShift(descriptor.getShift());
        state.setLastPostShift(descriptor.getPostShift());
        state.setTransformDecay(descriptor.getTransformDecay());
        return state;
    }

}
