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

import static com.amazon.randomcutforest.CommonUtils.toFloatArrayNullable;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.PredictorCorrector;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorMapper;
import com.amazon.randomcutforest.parkservices.state.returntypes.ComputeDescriptorMapper;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderMapper;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorState;
import com.amazon.randomcutforest.state.returntypes.DiVectorMapper;

@Getter
@Setter
public class ThresholdedRandomCutForestMapper
        implements IStateMapper<ThresholdedRandomCutForest, ThresholdedRandomCutForestState> {

    @Override
    public ThresholdedRandomCutForest toModel(ThresholdedRandomCutForestState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        Preprocessor preprocessor = preprocessorMapper.toModel(state.getPreprocessorStates()[0]);

        ForestMode forestMode = ForestMode.valueOf(state.getForestMode());
        TransformMethod transformMethod = TransformMethod.valueOf(state.getTransformMethod());

        ScoringStrategy scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;
        if (state.getScoringStrategy() != null && !state.getScoringStrategy().isEmpty()) {
            scoringStrategy = ScoringStrategy.valueOf(state.getScoringStrategy());
        }

        RCFComputeDescriptor descriptor;

        if (state.getLastDescriptorState() == null) {
            descriptor = new RCFComputeDescriptor(null, 0L);
            descriptor.setRCFScore(state.getLastAnomalyScore());
            descriptor.setInternalTimeStamp(state.getLastAnomalyTimeStamp());
            descriptor.setAttribution(new DiVectorMapper().toModel(state.getLastAnomalyAttribution()));
            descriptor.setRCFPoint(toFloatArrayNullable(state.getLastAnomalyPoint()));
            descriptor.setExpectedRCFPoint(toFloatArrayNullable(state.getLastExpectedPoint()));
            descriptor.setRelativeIndex(state.getLastRelativeIndex());
            descriptor.setScoringStrategy(scoringStrategy);
        } else {
            descriptor = new ComputeDescriptorMapper().toModel(state.getLastDescriptorState());
        }

        descriptor.setForestMode(forestMode);
        descriptor.setTransformMethod(transformMethod);
        descriptor.setScoringStrategy(scoringStrategy);
        descriptor
                .setImputationMethod(ImputationMethod.valueOf(state.getPreprocessorStates()[0].getImputationMethod()));

        PredictorCorrector predictorCorrector;
        if (state.getPredictorCorrectorState() == null) {
            BasicThresholderMapper thresholderMapper = new BasicThresholderMapper();
            BasicThresholder thresholder = thresholderMapper.toModel(state.getThresholderState());
            predictorCorrector = new PredictorCorrector(thresholder, preprocessor.getInputLength());
            predictorCorrector.setNumberOfAttributors(state.getNumberOfAttributors());
            predictorCorrector.setLastScore(new double[] { state.getLastScore() });
        } else {
            PredictorCorrectorMapper mapper = new PredictorCorrectorMapper();
            predictorCorrector = mapper.toModel(state.getPredictorCorrectorState());
        }

        return new ThresholdedRandomCutForest(forestMode, transformMethod, scoringStrategy, forest, predictorCorrector,
                preprocessor, descriptor);
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

        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();
        state.setPreprocessorStates(
                new PreprocessorState[] { preprocessorMapper.toState((Preprocessor) model.getPreprocessor()) });

        state.setPredictorCorrectorState(new PredictorCorrectorMapper().toState(model.getPredictorCorrector()));
        state.setForestMode(model.getForestMode().name());
        state.setTransformMethod(model.getTransformMethod().name());
        state.setScoringStrategy(model.getScoringStrategy().name());

        state.setLastDescriptorState(new ComputeDescriptorMapper().toState(model.getLastAnomalyDescriptor()));
        return state;
    }

}
