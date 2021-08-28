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

package com.amazon.randomcutforest.parkservices.threshold.state;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.parkservices.threshold.Deviation;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class BasicThresholderMapper implements IStateMapper<BasicThresholder, BasicThresholderState> {

    @Override
    public BasicThresholder toModel(BasicThresholderState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation primaryDeviation = deviationMapper.toModel(state.getPrimaryDeviationState());
        Deviation secondaryDeviation = deviationMapper.toModel(state.getSecondaryDeviationState());
        BasicThresholder thresholder = new BasicThresholder(primaryDeviation, secondaryDeviation);
        thresholder.setLowerThreshold(state.getLowerThreshold());
        thresholder.setUpperThreshold(state.getUpperThreshold());
        thresholder.setInitialThreshold(state.getInitialThreshold());
        thresholder.setElasticity(state.getElasticity());
        thresholder.setHorizon(state.getHorizon());
        thresholder.setCount(state.getCount());
        thresholder.setMinimumScores(state.getMinimumScores());
        thresholder.setAbsoluteScoreFraction(state.getAbsoluteScoreFraction());
        thresholder.setUpperZfactor(state.getUpperZfactor());
        thresholder.setZfactor(state.getZFactor());
        return thresholder;
    }

    @Override
    public BasicThresholderState toState(BasicThresholder model) {
        BasicThresholderState state = new BasicThresholderState();
        DeviationMapper deviationMapper = new DeviationMapper();

        state.setZFactor(model.getZFactor());
        state.setUpperZfactor(model.getUpperZfactor());
        state.setUpperThreshold(model.getUpperThreshold());
        state.setLowerThreshold(model.getLowerThreshold());
        state.setInitialThreshold(model.getInitialThreshold());
        state.setAbsoluteScoreFraction(model.getAbsoluteScoreFraction());
        state.setElasticity(model.getElasticity());
        state.setCount(model.getCount());
        state.setMinimumScores(model.getMinimumScores());
        state.setPrimaryDeviationState(deviationMapper.toState(model.getPrimaryDeviation()));
        state.setSecondaryDeviationState(deviationMapper.toState(model.getSecondaryDeviation()));
        state.setHorizon(model.getHorizon());
        return state;
    }

}
