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

package com.amazon.randomcutforest.parkservices.state.threshold;

import static com.amazon.randomcutforest.parkservices.state.statistics.DeviationMapper.getDeviations;
import static com.amazon.randomcutforest.parkservices.state.statistics.DeviationMapper.getStates;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.state.statistics.DeviationMapper;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class BasicThresholderMapper implements IStateMapper<BasicThresholder, BasicThresholderState> {

    @Override
    public BasicThresholder toModel(BasicThresholderState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation[] deviations = getDeviations(state.getDeviationStates(), deviationMapper);
        BasicThresholder thresholder = new BasicThresholder(deviations);
        thresholder.setAbsoluteThreshold(state.getAbsoluteThreshold());
        thresholder.setLowerThreshold(state.getLowerThreshold());
        thresholder.setInitialThreshold(state.getInitialThreshold());
        thresholder.setThresholdPersistence(state.getHorizon());
        thresholder.setCount(state.getCount());
        thresholder.setAutoThreshold(state.isAutoThreshold());
        thresholder.setMinimumScores(state.getMinimumScores());
        thresholder.setZfactor(state.getZFactor());
        return thresholder;
    }

    @Override
    public BasicThresholderState toState(BasicThresholder model) {
        BasicThresholderState state = new BasicThresholderState();
        DeviationMapper deviationMapper = new DeviationMapper();

        state.setZFactor(model.getZFactor());
        state.setLowerThreshold(model.getLowerThreshold());
        state.setAbsoluteThreshold(model.getAbsoluteThreshold());
        state.setInitialThreshold(model.getInitialThreshold());
        state.setCount(model.getCount());
        state.setAutoThreshold(model.isAutoThreshold());
        state.setMinimumScores(model.getMinimumScores());
        state.setDeviationStates(getStates(model.getDeviations(), deviationMapper));
        state.setHorizon(model.getThresholdPersistence());
        return state;
    }

}
