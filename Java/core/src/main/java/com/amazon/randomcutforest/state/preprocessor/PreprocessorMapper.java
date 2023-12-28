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

package com.amazon.randomcutforest.state.preprocessor;

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.state.statistics.DeviationMapper.getDeviations;
import static com.amazon.randomcutforest.state.statistics.DeviationMapper.getStates;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.statistics.DeviationMapper;
import com.amazon.randomcutforest.statistics.Deviation;

@Getter
@Setter
public class PreprocessorMapper implements IStateMapper<Preprocessor, PreprocessorState> {

    @Override
    public Preprocessor toModel(PreprocessorState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation[] deviations = getDeviations(state.getDeviationStates(), deviationMapper);
        Deviation[] timeStampDeviations = getDeviations(state.getTimeStampDeviationStates(), deviationMapper);
        Deviation[] dataQuality = getDeviations(state.getDataQualityStates(), deviationMapper);
        Preprocessor.Builder<?> preprocessorBuilder = new Preprocessor.Builder<>()
                .forestMode(ForestMode.valueOf(state.getForestMode())).shingleSize(state.getShingleSize())
                .dimensions(state.getDimensions()).normalizeTime(state.isNormalizeTime())
                .imputationMethod(ImputationMethod.valueOf(state.getImputationMethod()))
                .fillValues(state.getDefaultFill()).inputLength(state.getInputLength()).weights(state.getWeights())
                .transformMethod(TransformMethod.valueOf(state.getTransformMethod()))
                .startNormalization(state.getStartNormalization()).useImputedFraction(state.getUseImputedFraction())
                .timeDeviations(timeStampDeviations).deviations(deviations).dataQuality(dataQuality)
                .transformDecay(state.getTimeDecay());

        Preprocessor preprocessor = preprocessorBuilder.build();
        preprocessor.setInitialValues(state.getInitialValues());
        preprocessor.setInitialTimeStamps(state.getInitialTimeStamps());
        preprocessor.setClipFactor(state.getClipFactor());
        preprocessor.setValuesSeen(state.getValuesSeen());
        preprocessor.setInternalTimeStamp(state.getInternalTimeStamp());
        preprocessor.setLastShingledInput(state.getLastShingledInput());
        preprocessor.setLastShingledPoint(toFloatArray(state.getLastShingledPoint()));
        preprocessor.setPreviousTimeStamps(state.getPreviousTimeStamps());
        preprocessor.setNormalizeTime(state.isNormalizeTime());
        preprocessor.setFastForward(state.isFastForward());
        return preprocessor;
    }

    @Override
    public PreprocessorState toState(Preprocessor model) {
        PreprocessorState state = new PreprocessorState();
        state.setShingleSize(model.getShingleSize());
        state.setDimensions(model.getDimension());
        state.setInputLength(model.getInputLength());
        state.setClipFactor(model.getClipFactor());
        state.setDefaultFill(model.getDefaultFill());
        state.setImputationMethod(model.getImputationMethod().name());
        state.setTransformMethod(model.getTransformMethod().name());
        state.setWeights(model.getWeights());
        state.setForestMode(model.getMode().name());
        state.setInitialTimeStamps(model.getInitialTimeStamps());
        state.setInitialValues(model.getInitialValues());
        state.setUseImputedFraction(model.getUseImputedFraction());
        state.setNormalizeTime(model.isNormalizeTime());
        state.setStartNormalization(model.getStartNormalization());
        state.setStopNormalization(model.getStopNormalization());
        state.setPreviousTimeStamps(model.getPreviousTimeStamps());
        state.setLastShingledInput(model.getLastShingledInput());
        state.setLastShingledPoint(toDoubleArray(model.getLastShingledPoint()));
        state.setValuesSeen(model.getValuesSeen());
        state.setInternalTimeStamp(model.getInternalTimeStamp());
        DeviationMapper deviationMapper = new DeviationMapper();
        state.setTimeDecay(model.getTransformDecay());
        state.setDeviationStates(getStates(model.getDeviationList(), deviationMapper));
        state.setTimeStampDeviationStates(getStates(model.getTimeStampDeviations(), deviationMapper));
        state.setDataQualityStates(getStates(model.getDataQuality(), deviationMapper));
        state.setFastForward(model.isFastForward());
        return state;
    }

}
