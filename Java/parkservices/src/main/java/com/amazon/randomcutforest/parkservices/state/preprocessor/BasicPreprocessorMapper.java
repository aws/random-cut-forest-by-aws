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

import com.amazon.randomcutforest.parkservices.preprocessor.BasicPreprocessor;
import com.amazon.randomcutforest.parkservices.state.statistics.DeviationMapper;
import com.amazon.randomcutforest.parkservices.state.statistics.DeviationState;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class BasicPreprocessorMapper implements IStateMapper<BasicPreprocessor, BasicPreprocessorState> {

    @Override
    public BasicPreprocessor toModel(BasicPreprocessorState state, long seed) {
        DeviationMapper deviationMapper = new DeviationMapper();
        Deviation timeStampDeviation = deviationMapper.toModel(state.getTimeStampDeviationState());
        Deviation[] deviations = null;
        if (state.getDeviationStates() != null) {
            deviations = new Deviation[state.getDeviationStates().length];
            for (int i = 0; i < state.getDeviationStates().length; i++) {
                deviations[i] = deviationMapper.toModel(state.getDeviationStates()[i]);
            }
        }
        BasicPreprocessor.Builder<?> preprocessorBuilder = new BasicPreprocessor.Builder<>()
                .setMode(state.getForestMode()).shingleSize(state.getShingleSize()).dimensions(state.getDimensions())
                .fillIn(state.getImputationMethod()).fillValues(state.getDefaultFill())
                .inputLength(state.getInputLength()).transformMethod(state.getTransformMethod())
                .startNormalization(state.getStartNormalization()).useImputedFraction(state.getUseImputedFraction())
                .timeDeviation(timeStampDeviation);

        if (deviations != null) {
            preprocessorBuilder.deviations(deviations);
        }

        BasicPreprocessor preprocessor = preprocessorBuilder.build();
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
    public BasicPreprocessorState toState(BasicPreprocessor model) {
        BasicPreprocessorState state = new BasicPreprocessorState();
        state.setShingleSize(model.getShingleSize());
        state.setDimensions(model.getDimension());
        state.setInputLength(model.getInputLength());
        state.setClipFactor(model.getClipFactor());
        state.setDefaultFill(model.getDefaultFill());
        state.setImputationMethod(model.getImputationMethod());
        state.setTransformMethod(model.getTransformMethod());
        state.setForestMode(model.getMode());
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
        DeviationState[] deviationStates = null;
        if (model.getDeviationList() != null) {
            Deviation[] list = model.getDeviationList();
            deviationStates = new DeviationState[list.length];
            for (int i = 0; i < list.length; i++) {
                deviationStates[i] = deviationMapper.toState(list[i]);
            }
        }
        return state;
    }

}
