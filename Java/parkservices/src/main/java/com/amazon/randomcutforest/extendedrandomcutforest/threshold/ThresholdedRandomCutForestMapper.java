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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.extendedrandomcutforest.threshold.state.BasicThresholderMapper;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;

@Getter
@Setter
public class ThresholdedRandomCutForestMapper implements IStateMapper<ThresholdedRandomCutForest, ThresholdedRandomCutForestState> {

    @Override
    public ThresholdedRandomCutForest toModel(ThresholdedRandomCutForestState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        BasicThresholderMapper thresholderMapper = new BasicThresholderMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        BasicThresholder thresholder = thresholderMapper.toModel(state.getThresholderState());
        return  new ThresholdedRandomCutForest(forest, thresholder, state);
    }

    @Override
    public ThresholdedRandomCutForestState toState(ThresholdedRandomCutForest model) {
        ThresholdedRandomCutForestState state = new ThresholdedRandomCutForestState();
        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        randomCutForestMapper.setPartialTreeStateEnabled(true);
        randomCutForestMapper.setSaveTreeStateEnabled(true);
        randomCutForestMapper.setCompressionEnabled(true);
        randomCutForestMapper.setSaveCoordinatorStateEnabled(true);
        randomCutForestMapper.setSaveExecutorContextEnabled(true);

        state.setForestState(randomCutForestMapper.toState(model.getForest()));

        BasicThresholderMapper thresholderMapper =  new BasicThresholderMapper();
        state.setThresholderState(thresholderMapper.toState((BasicThresholder) model.getThresholder()));


        state.setTriggerFactor(model.getTriggerFactor());
        state.setInAnomaly(model.isInHighScoreRegion());
        state.setLastScore(model.getLastScore());
        state.setLastAnomalyTimeStamp(model.getLastAnomalyTimeStamp());
        state.setLastAnomalyScore(model.getLastAnomalyScore());
        state.setLastAnomalyAttribution(model.getLastAnomalyAttribution());
        state.setLastAnomalyPoint(model.getLastAnomalyPoint());
        state.setLastExpectedPoint(model.getLastExpectedPoint());
        state.setIgnoreSimilar(model.isIgnoreSimilar());
        state.setIgnoreSimilarFactor(model.getIgnoreSimilarFactor());
        state.setPreviousIsPotentialAnomaly(model.isPreviousIsPotentialAnomaly());
        state.setNumberOfAttributors(model.getNumberOfAttributors());

        return state;
    }


}
