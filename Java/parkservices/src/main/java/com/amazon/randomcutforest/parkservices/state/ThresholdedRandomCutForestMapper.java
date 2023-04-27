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

package com.amazon.randomcutforest.parkservices.state;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.IRCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.PredictorCorrector;
import com.amazon.randomcutforest.parkservices.RCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.parkservices.state.preprocessor.PreprocessorState;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderMapper;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.returntypes.DiVectorMapper;

@Getter
@Setter
public class ThresholdedRandomCutForestMapper
        implements IStateMapper<ThresholdedRandomCutForest, ThresholdedRandomCutForestState> {

    @Override
    public ThresholdedRandomCutForest toModel(ThresholdedRandomCutForestState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        BasicThresholderMapper thresholderMapper = new BasicThresholderMapper();
        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        BasicThresholder thresholder = thresholderMapper.toModel(state.getThresholderState());
        Preprocessor preprocessor = preprocessorMapper.toModel(state.getPreprocessorStates()[0]);

        ForestMode forestMode = ForestMode.valueOf(state.getForestMode());
        TransformMethod transformMethod = TransformMethod.valueOf(state.getTransformMethod());

        RCFComputeDescriptor descriptor = new RCFComputeDescriptor(null, 0L);
        descriptor.setRCFScore(state.getLastAnomalyScore());
        descriptor.setInternalTimeStamp(state.getLastAnomalyTimeStamp());
        descriptor.setAttribution(new DiVectorMapper().toModel(state.getLastAnomalyAttribution()));
        descriptor.setRCFPoint(state.getLastAnomalyPoint());
        descriptor.setExpectedRCFPoint(state.getLastExpectedPoint());
        descriptor.setRelativeIndex(state.getLastRelativeIndex());
        descriptor.setForestMode(forestMode);
        descriptor.setTransformMethod(transformMethod);
        descriptor
                .setImputationMethod(ImputationMethod.valueOf(state.getPreprocessorStates()[0].getImputationMethod()));

        PredictorCorrector predictorCorrector = new PredictorCorrector(thresholder);
        predictorCorrector.setIgnoreSimilar(state.isIgnoreSimilar());
        predictorCorrector.setIgnoreSimilarFactor(state.getIgnoreSimilarFactor());
        predictorCorrector.setTriggerFactor(state.getTriggerFactor());
        predictorCorrector.setNumberOfAttributors(state.getNumberOfAttributors());
        predictorCorrector.setLastScore(state.getLastScore());

        return new ThresholdedRandomCutForest(forestMode, transformMethod, forest, predictorCorrector, preprocessor,
                descriptor);
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
        state.setThresholderState(thresholderMapper.toState(model.getThresholder()));

        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();
        state.setPreprocessorStates(
                new PreprocessorState[] { preprocessorMapper.toState((Preprocessor) model.getPreprocessor()) });
        state.setTriggerFactor(model.getPredictorCorrector().getTriggerFactor());
        state.setIgnoreSimilar(model.getPredictorCorrector().isIgnoreSimilar());
        state.setIgnoreSimilarFactor(model.getPredictorCorrector().getIgnoreSimilarFactor());
        state.setNumberOfAttributors(model.getPredictorCorrector().getNumberOfAttributors());
        state.setForestMode(model.getForestMode().name());
        state.setTransformMethod(model.getTransformMethod().name());

        IRCFComputeDescriptor descriptor = model.getLastAnomalyDescriptor();
        state.setLastAnomalyTimeStamp(descriptor.getInternalTimeStamp());
        state.setLastAnomalyScore(descriptor.getRCFScore());
        state.setLastAnomalyAttribution(new DiVectorMapper().toState(descriptor.getAttribution()));
        state.setLastAnomalyPoint(descriptor.getRCFPoint());
        state.setLastExpectedPoint(descriptor.getExpectedRCFPoint());
        state.setLastRelativeIndex(descriptor.getRelativeIndex());
        state.setLastScore(model.getLastScore());

        return state;
    }

}
