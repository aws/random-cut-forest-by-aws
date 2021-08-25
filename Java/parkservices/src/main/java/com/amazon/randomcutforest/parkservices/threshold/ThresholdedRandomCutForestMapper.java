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

package com.amazon.randomcutforest.parkservices.threshold;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.parkservices.state.DiVectorMapper;
import com.amazon.randomcutforest.parkservices.threshold.state.BasicThresholderMapper;
import com.amazon.randomcutforest.parkservices.threshold.state.DeviationMapper;
import com.amazon.randomcutforest.parkservices.threshold.state.DeviationState;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;

@Getter
@Setter
public class ThresholdedRandomCutForestMapper
        implements IStateMapper<ThresholdedRandomCutForest, ThresholdedRandomCutForestState> {

    @Override
    public ThresholdedRandomCutForest toModel(ThresholdedRandomCutForestState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        BasicThresholderMapper thresholderMapper = new BasicThresholderMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        BasicThresholder thresholder = thresholderMapper.toModel(state.getThresholderState());
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation timeStampDeviation = deviationMapper.toModel(state.getTimeStampDeviationState());
        Deviation[] deviations = null;
        if (state.isNormalizeValues()) {
            deviations = new Deviation[state.getDeviationStates().length];
            for (int i = 0; i < state.getDeviationStates().length; i++) {
                deviations[i] = deviationMapper.toModel(state.getDeviationStates()[i]);
            }
        }
        ThresholdedRandomCutForest tForest = new ThresholdedRandomCutForest(forest, thresholder, timeStampDeviation,
                deviations);
        tForest.setIgnoreSimilar(state.isIgnoreSimilar());
        tForest.setIgnoreSimilarFactor(state.getIgnoreSimilarFactor());
        tForest.setLastScore(state.getLastScore());
        tForest.setLastAnomalyScore(state.getLastAnomalyScore());
        tForest.setLastAnomalyTimeStamp(state.getLastAnomalyTimeStamp());
        tForest.setPreviousIsPotentialAnomaly(state.isPreviousIsPotentialAnomaly());
        tForest.setTriggerFactor(state.getTriggerFactor());
        tForest.setNumberOfAttributors(state.getNumberOfAttributors());
        tForest.setInHighScoreRegion(state.isInHighScoreRegion());
        tForest.setLastAnomalyAttribution(new DiVectorMapper().toModel(state.getLastAnomalyAttribution()));
        tForest.setLastAnomalyPoint(state.getLastAnomalyPoint());
        tForest.setLastExpectedPoint(state.getLastExpectedPoint());
        tForest.setValuesSeen(state.getValuesSeen());
        tForest.setPreviousTimeStamps(state.getPreviousTimeStamps());
        tForest.setImputationMethod(state.getImputationMethod());
        tForest.setLastShingledPoint(state.getLastShingledPoint());
        tForest.setNormalizeTime(state.isNormalizeTime());
        tForest.setDefaultFill(state.getDefaultFill());
        tForest.setUseImputedFraction(state.getUseImputedFraction());
        tForest.setNormalizeValues(state.isNormalizeValues());
        tForest.setStopNormalization(state.getStopNormalization());
        tForest.setClipFactor(state.getClipFactor());
        tForest.setForestMode(state.getForestMode());
        tForest.setLastShingledInput(state.getLastShingledInput());
        tForest.setDifferencing(state.isDifferencing());
        tForest.setLastRelativeIndex(state.getLastRelativeIndex());
        tForest.setLastReset(state.getLastReset());
        return tForest;
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

        BasicThresholderMapper thresholderMapper = new BasicThresholderMapper();
        state.setThresholderState(thresholderMapper.toState((BasicThresholder) model.getThresholder()));

        DeviationMapper deviationMapper = new DeviationMapper();
        state.setTimeStampDeviationState(deviationMapper.toState(model.getTimeStampDeviation()));
        state.setNormalizeValues(model.isNormalizeValues());
        DeviationState[] deviationStates = null;
        if (model.isNormalizeValues()) {
            deviationStates = new DeviationState[model.getDeviationList().length];
            for (int i = 0; i < model.deviationList.length; i++) {
                deviationStates[i] = deviationMapper.toState(model.deviationList[i]);
            }
        }

        state.setTriggerFactor(model.getTriggerFactor());
        state.setInHighScoreRegion(model.isInHighScoreRegion());
        state.setLastScore(model.getLastScore());
        state.setLastAnomalyTimeStamp(model.getLastAnomalyTimeStamp());
        state.setLastAnomalyScore(model.getLastAnomalyScore());
        state.setLastAnomalyAttribution(new DiVectorMapper().toState(model.getLastAnomalyAttribution()));
        state.setLastAnomalyPoint(model.getLastAnomalyPoint());
        state.setLastExpectedPoint(model.getLastExpectedPoint());
        state.setIgnoreSimilar(model.isIgnoreSimilar());
        state.setPreviousTimeStamps(model.getPreviousTimeStamps());
        state.setValuesSeen(model.getValuesSeen());
        state.setIgnoreSimilarFactor(model.getIgnoreSimilarFactor());
        state.setPreviousIsPotentialAnomaly(model.isPreviousIsPotentialAnomaly());
        state.setNumberOfAttributors(model.getNumberOfAttributors());
        state.setImputationMethod(model.getImputationMethod());
        state.setForestMode(model.getForestMode());
        state.setLastShingledPoint(model.getLastShingledPoint());
        state.setDefaultFill(model.getDefaultFill());
        state.setNormalizeTime(model.isNormalizeTime());
        state.setUseImputedFraction(model.getUseImputedFraction());
        state.setClipFactor(model.getClipFactor());
        state.setStopNormalization(model.getStopNormalization());
        state.setLastShingledInput(model.getLastShingledInput());
        state.setDifferencing(model.isDifferencing());
        state.setLastRelativeIndex(model.getLastRelativeIndex());
        state.setLastReset(model.getLastReset());
        return state;
    }

}
