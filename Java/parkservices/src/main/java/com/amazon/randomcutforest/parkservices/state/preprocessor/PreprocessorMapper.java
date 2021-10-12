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

package com.amazon.randomcutforest.parkservices.state.preprocessor;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.state.statistics.DeviationMapper;
import com.amazon.randomcutforest.parkservices.state.statistics.DeviationState;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class PreprocessorMapper implements IStateMapper<Preprocessor, PreprocessorState> {

    @Override
    public Preprocessor toModel(PreprocessorState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation timeStampDeviation = deviationMapper.toModel(state.getTimeStampDeviationState());
        Deviation dataQuality = deviationMapper.toModel(state.getDataQualityState());
        Deviation[] deviations = null;
        if (state.getDeviationStates() != null) {
            deviations = new Deviation[state.getDeviationStates().length];
            for (int i = 0; i < state.getDeviationStates().length; i++) {
                deviations[i] = deviationMapper.toModel(state.getDeviationStates()[i]);
            }
        }
        Preprocessor.Builder<?> preprocessorBuilder = new Preprocessor.Builder<>()
                .forestMode(ForestMode.valueOf(state.getForestMode())).shingleSize(state.getShingleSize())
                .dimensions(state.getDimensions())
                .imputationMethod(ImputationMethod.valueOf(state.getImputationMethod()))
                .fillValues(state.getDefaultFill()).inputLength(state.getInputLength()).weights(state.getWeights())
                .transformMethod(TransformMethod.valueOf(state.getTransformMethod()))
                .startNormalization(state.getStartNormalization()).useImputedFraction(state.getUseImputedFraction())
                .timeDeviation(timeStampDeviation).dataQuality(dataQuality);

        if (deviations != null) {
            preprocessorBuilder.deviations(deviations);
        }

        Preprocessor preprocessor = preprocessorBuilder.build();
        preprocessor.setInitialValues(state.getInitialValues());
        preprocessor.setInitialTimeStamps(state.getInitialTimeStamps());
        preprocessor.setClipFactor(state.getClipFactor());
        preprocessor.setValuesSeen(state.getValuesSeen());
        preprocessor.setInternalTimeStamp(state.getInternalTimeStamp());
        preprocessor.setLastShingledInput(state.getLastShingledInput());
        preprocessor.setLastShingledPoint(state.getLastShingledPoint());
        preprocessor.setPreviousTimeStamps(state.getPreviousTimeStamps());
        preprocessor.setNormalizeTime(state.isNormalizeTime());
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
        state.setLastShingledPoint(model.getLastShingledPoint());
        state.setValuesSeen(model.getValuesSeen());
        state.setInternalTimeStamp(model.getInternalTimeStamp());
        DeviationMapper deviationMapper = new DeviationMapper();
        state.setTimeStampDeviationState(deviationMapper.toState(model.getTimeStampDeviation()));
        state.setDataQualityState(deviationMapper.toState(model.getDataQuality()));
        DeviationState[] deviationStates = null;
        if (model.getDeviationList() != null) {
            Deviation[] list = model.getDeviationList();
            deviationStates = new DeviationState[list.length];
            for (int i = 0; i < list.length; i++) {
                deviationStates[i] = deviationMapper.toState(list[i]);
            }
        }
        state.setDeviationStates(deviationStates);
        return state;
    }

}
