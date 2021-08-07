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

package com.amazon.randomcutforest.threshold.state;

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.threshold.CorrectorThresholder;
import com.amazon.randomcutforest.threshold.Deviation;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class CorrectorThresholderMapper implements IStateMapper<CorrectorThresholder, CorrectorThresholderState> {

    @Override
    public CorrectorThresholder toModel(CorrectorThresholderState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation simpleDeviation = deviationMapper.toModel(state.getSimpleDeviationState());
        Deviation scoreDiff = deviationMapper.toModel(state.getScoreDiffState());
        return new CorrectorThresholder(state, simpleDeviation, scoreDiff);
    }

    @Override
    public CorrectorThresholderState toState(CorrectorThresholder model) {
        CorrectorThresholderState state = new CorrectorThresholderState();
        DeviationMapper deviationMapper = new DeviationMapper();

        state.setZ_FACTOR(model.getzFactor());
        state.setUPPER_ANCHOR_SCORE(model.getUpperThreshold());
        state.setTRIGGER_FACTOR(model.getTriggerFactor());
        state.setDiscount(model.getDiscount());
        state.setElasticity(model.getElasticity());
        state.setBASIC_FACTOR(model.getBASIC_FACTOR());
        state.setAbsoluteThreshold(model.getLowerThreshold());
        state.setCount(model.getCount());
        state.setInAnomaly(model.isInAnomaly());
        state.setBaseDimension(model.getBaseDimension());
        state.setShingleSize(model.getShingleSize());
        state.setElasticity(model.getElasticity());
        state.setMinimumScores(model.getMinimumScores());
        state.setAttributionEnabled(model.isAttributionEnabled());
        state.setSimpleDeviationState(deviationMapper.toState(model.getSimpleDeviation()));
        state.setLastScore(model.getLastScore());
        state.setLastAnomalyTimeStamp(model.getLastAnomalyTimeStamp());
        state.setLastAnomalyScore(model.getLastAnomalyScore());
        state.setLastAnomalyAttribution(model.getLastAnomalyAttribution());
        state.setIgnoreSimilar(model.isIgnoreSimilar());
        state.setPreviousIsPotentialAnomaly(model.isPreviousIsPotentialAnomaly());
        state.setScoreDiffState(deviationMapper.toState(model.getScoreDiff()));
        return state;
    }

}
