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
import com.amazon.randomcutforest.parkservices.PredictorCorrector;
import com.amazon.randomcutforest.parkservices.RCFCaster;
import com.amazon.randomcutforest.parkservices.calibration.ErrorHandler;
import com.amazon.randomcutforest.parkservices.config.Calibration;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.state.errorhandler.ErrorHandlerMapper;
import com.amazon.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorMapper;
import com.amazon.randomcutforest.parkservices.state.returntypes.ComputeDescriptorMapper;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorState;

@Getter
@Setter
public class RCFCasterMapper implements IStateMapper<RCFCaster, RCFCasterState> {

    @Override
    public RCFCasterState toState(RCFCaster model) {
        RCFCasterState state = new RCFCasterState();

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
        state.setLastDescriptorState(
                new ComputeDescriptorMapper().toState((RCFComputeDescriptor) model.getLastAnomalyDescriptor()));
        state.setForestMode(model.getForestMode().name());
        state.setTransformMethod(model.getTransformMethod().name());

        state.setForecastHorizon(model.getForecastHorizon());

        ErrorHandlerMapper errorHandlerMapper = new ErrorHandlerMapper();
        state.setErrorHandler(errorHandlerMapper.toState(model.getErrorHandler()));

        state.setErrorHorizon(model.getErrorHorizon());
        state.setCalibrationMethod(model.getCalibrationMethod().name());
        state.setScoringStrategy(model.getScoringStrategy().name());
        return state;
    }

    @Override
    public RCFCaster toModel(RCFCasterState state, long seed) {
        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        Preprocessor preprocessor = preprocessorMapper.toModel(state.getPreprocessorStates()[0]);

        ForestMode forestMode = ForestMode.valueOf(state.getForestMode());
        TransformMethod transformMethod = TransformMethod.valueOf(state.getTransformMethod());

        RCFComputeDescriptor descriptor = new ComputeDescriptorMapper().toModel(state.getLastDescriptorState());
        descriptor.setForestMode(forestMode);
        descriptor.setTransformMethod(transformMethod);
        descriptor
                .setImputationMethod(ImputationMethod.valueOf(state.getPreprocessorStates()[0].getImputationMethod()));
        descriptor.setShingleSize(preprocessor.getShingleSize());

        PredictorCorrectorMapper mapper = new PredictorCorrectorMapper();
        PredictorCorrector predictorCorrector = mapper.toModel(state.getPredictorCorrectorState());

        ErrorHandlerMapper errorHandlerMapper = new ErrorHandlerMapper();
        ErrorHandler errorHandler = errorHandlerMapper.toModel(state.getErrorHandler());

        Calibration calibrationMethod = Calibration.valueOf(state.getCalibrationMethod());
        ScoringStrategy scoringStrategy = ScoringStrategy.valueOf(state.getScoringStrategy());

        return new RCFCaster(forestMode, transformMethod, scoringStrategy, forest, predictorCorrector, preprocessor,
                descriptor, state.getForecastHorizon(), errorHandler, state.getErrorHorizon(), calibrationMethod);
    }

}
