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

package com.amazon.randomcutforest.parkservices.state.predictorcorrector;

import com.amazon.randomcutforest.parkservices.PredictorCorrector;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.state.returntypes.ComputeDescriptorMapper;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderMapper;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderState;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.statistics.DeviationMapper;
import com.amazon.randomcutforest.state.statistics.DeviationState;
import com.amazon.randomcutforest.statistics.Deviation;

public class PredictorCorrectorMapper implements IStateMapper<PredictorCorrector, PredictorCorrectorState> {

    @Override
    public PredictorCorrectorState toState(PredictorCorrector model) {
        PredictorCorrectorState state = new PredictorCorrectorState();
        state.setLastScore(model.getLastScore());
        state.setNumberOfAttributors(model.getNumberOfAttributors());
        state.setIgnoreNearExpected(model.getIgnoreNearExpected());
        BasicThresholderMapper mapper = new BasicThresholderMapper();
        BasicThresholder[] thresholders = model.getThresholders();
        BasicThresholderState thresholderState[] = new BasicThresholderState[thresholders.length];
        for (int y = 0; y < thresholders.length; y++) {
            thresholderState[y] = mapper.toState(thresholders[y]);
        }
        state.setThresholderStates(thresholderState);
        DeviationMapper devMapper = new DeviationMapper();
        Deviation[] deviations = model.getDeviations();
        state.setAutoAdjust(model.isAutoAdjust());
        if (state.isAutoAdjust()) {
            DeviationState deviationState[] = new DeviationState[deviations.length];
            for (int y = 0; y < deviations.length; y++) {
                deviationState[y] = devMapper.toState(deviations[y]);
            }
            state.setDeviationStates(deviationState);
        }
        state.setNoiseFactor(model.getNoiseFactor());
        state.setBaseDimension(model.getBaseDimension());
        state.setLastStrategy(model.getLastStrategy().name());
        state.setRandomSeed(model.getRandomSeed());
        if (model.getLastDescriptor() != null) {
            ComputeDescriptorMapper descriptorMapper = new ComputeDescriptorMapper();
            state.setLastDescriptor(descriptorMapper.toState(model.getLastDescriptor()));
        }
        state.setModeInformation(model.getModeInformation());
        state.setRunLength(model.getRunLength());
        state.setIgnoreDrift(model.isIgnoreDrift());
        state.setSamplingSuppport(model.getSamplingSupport());
        return state;
    }

    @Override
    public PredictorCorrector toModel(PredictorCorrectorState state, long seed) {
        BasicThresholderMapper mapper = new BasicThresholderMapper();
        int num = state.getThresholderStates().length;
        BasicThresholder[] thresholders = new BasicThresholder[num];
        for (int i = 0; i < num; i++) {
            thresholders[i] = mapper.toModel(state.getThresholderStates()[i]);
        }
        Deviation[] deviations = null;
        if (state.isAutoAdjust()) {
            DeviationMapper devMapper = new DeviationMapper();
            deviations = new Deviation[state.getDeviationStates().length];
            for (int y = 0; y < deviations.length; y++) {
                deviations[y] = devMapper.toModel(state.getDeviationStates()[y]);
            }
        }
        PredictorCorrector predictorCorrector = new PredictorCorrector(thresholders, deviations,
                state.getBaseDimension(), state.getRandomSeed());
        predictorCorrector.setNumberOfAttributors(state.getNumberOfAttributors());
        predictorCorrector.setLastStrategy(ScoringStrategy.valueOf(state.getLastStrategy()));
        predictorCorrector.setLastScore(state.getLastScore());
        predictorCorrector.setIgnoreNearExpected(state.getIgnoreNearExpected());
        predictorCorrector.setAutoAdjust(state.isAutoAdjust());
        predictorCorrector.setNoiseFactor(state.getNoiseFactor());
        predictorCorrector.setRunLength(state.getRunLength());
        predictorCorrector.setModeInformation(state.getModeInformation());
        if (state.getLastDescriptor() != null) {
            ComputeDescriptorMapper descriptorMapper = new ComputeDescriptorMapper();
            predictorCorrector.setLastDescriptor(descriptorMapper.toModel(state.getLastDescriptor()));
        }
        predictorCorrector.setIgnoreDrift(state.isIgnoreDrift());
        predictorCorrector.setSamplingSupport(state.getSamplingSuppport());
        return predictorCorrector;
    }

}
