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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold.state;

import com.amazon.randomcutforest.extendedrandomcutforest.threshold.BasicThresholder;
import com.amazon.randomcutforest.extendedrandomcutforest.threshold.Deviation;
import com.amazon.randomcutforest.state.IStateMapper;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class BasicThresholderMapper implements IStateMapper<BasicThresholder, BasicThresholderState> {

    @Override
    public BasicThresholder toModel(BasicThresholderState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation primaryDeviation = deviationMapper.toModel(state.getPrimaryDeviationState());
        Deviation secondaryDeviation = deviationMapper.toModel(state.getSecondaryDeviationState());
        return new BasicThresholder(state, primaryDeviation,secondaryDeviation);
    }

    @Override
    public BasicThresholderState toState(BasicThresholder model) {
        BasicThresholderState state = new BasicThresholderState();
        DeviationMapper deviationMapper = new DeviationMapper();

        state.setZFactor(model.getzFactor());
        state.setUpperZfactor(model.getUpperZFactor());
        state.setUpperThreshold(model.getUpperThreshold());
        state.setLowerThreshold(model.getLowerThreshold());
        state.setInitialThreshold(model.getInitialThreshold());
        state.setAbsoluteScoreFraction(model.getAbsoluteScoreFraction());
        state.setPrimaryDiscount(model.getPrimaryDiscount());
        state.setSecondaryDiscount(model.getSecondaryDiscount());
        state.setElasticity(model.getElasticity());
        state.setCount(model.getCount());
        state.setElasticity(model.getElasticity());
        state.setMinimumScores(model.getMinimumScores());
        state.setPrimaryDeviationState(deviationMapper.toState(model.getPrimaryDeviation()));
        state.setSecondaryDeviationState(deviationMapper.toState(model.getSecondaryDeviation()));
        state.setLastScore(model.getLastScore());
        state.setHorizon(model.getHorizon());
        return state;
    }

}
