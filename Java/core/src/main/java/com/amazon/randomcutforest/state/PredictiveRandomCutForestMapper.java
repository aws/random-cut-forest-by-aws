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

package com.amazon.randomcutforest.state;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.PredictiveRandomCutForest;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorState;

@Getter
@Setter
public class PredictiveRandomCutForestMapper
        implements IStateMapper<PredictiveRandomCutForest, PredictiveRandomCutForestState> {

    @Override
    public PredictiveRandomCutForest toModel(PredictiveRandomCutForestState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        PreprocessorMapper preprocessorMapper = new PreprocessorMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        Preprocessor preprocessor = preprocessorMapper.toModel(state.getPreprocessorStates()[0]);

        ForestMode forestMode = ForestMode.valueOf(state.getForestMode());
        TransformMethod transformMethod = TransformMethod.valueOf(state.getTransformMethod());

        return new PredictiveRandomCutForest(forestMode, transformMethod, preprocessor, forest);
    }

    @Override
    public PredictiveRandomCutForestState toState(PredictiveRandomCutForest model) {
        PredictiveRandomCutForestState state = new PredictiveRandomCutForestState();
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

        state.setForestMode(model.getForestMode().name());
        state.setTransformMethod(model.getTransformMethod().name());
        return state;
    }

}
