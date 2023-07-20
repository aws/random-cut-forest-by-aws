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

import com.amazon.randomcutforest.config.CorrectionMode;
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
        descriptor.setRCFScore(state.getScore());
        descriptor.setInternalTimeStamp(state.getInternalTimeStamp());
        descriptor.setAttribution(new DiVectorMapper().toModel(state.getAttribution()));
        descriptor.setRCFPoint(state.getPoint());
        descriptor.setExpectedRCFPoint(state.getExpectedPoint());
        descriptor.setRelativeIndex(state.getRelativeIndex());
        descriptor.setScoringStrategy(ScoringStrategy.valueOf(state.getStrategy()));
        descriptor.setShift(state.getShift());
        descriptor.setPostShift(state.getPostShift());
        descriptor.setTransformDecay(state.getTransformDecay());
        descriptor.setPostDeviations(state.getPostDeviations());
        descriptor.setScale(state.getScale());
        descriptor.setAnomalyGrade(state.getAnomalyGrade());
        descriptor.setThreshold(state.getThreshold());
        descriptor.setCorrectionMode(CorrectionMode.valueOf(state.getCorrectionMode()));
        return descriptor;
    }

    @Override
    public ComputeDescriptorState toState(RCFComputeDescriptor descriptor) {

        ComputeDescriptorState state = new ComputeDescriptorState();
        state.setInternalTimeStamp(descriptor.getInternalTimeStamp());
        state.setScore(descriptor.getRCFScore());
        state.setAttribution(new DiVectorMapper().toState(descriptor.getAttribution()));
        state.setPoint(descriptor.getRCFPoint());
        state.setExpectedPoint(descriptor.getExpectedRCFPoint());
        state.setRelativeIndex(descriptor.getRelativeIndex());
        state.setStrategy(descriptor.getScoringStrategy().name());
        state.setShift(descriptor.getShift());
        state.setPostShift(descriptor.getPostShift());
        state.setTransformDecay(descriptor.getTransformDecay());
        state.setPostDeviations(descriptor.getPostDeviations());
        state.setScale(descriptor.getScale());
        state.setAnomalyGrade(descriptor.getAnomalyGrade());
        state.setThreshold(descriptor.getThreshold());
        state.setCorrectionMode(descriptor.getCorrectionMode().name());
        return state;
    }

}
