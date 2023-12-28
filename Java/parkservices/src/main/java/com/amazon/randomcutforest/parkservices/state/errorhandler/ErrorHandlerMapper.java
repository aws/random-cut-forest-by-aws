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

package com.amazon.randomcutforest.parkservices.state.errorhandler;

import static com.amazon.randomcutforest.state.statistics.DeviationMapper.getDeviations;
import static com.amazon.randomcutforest.state.statistics.DeviationMapper.getStates;

import com.amazon.randomcutforest.PredictiveRandomCutForest;
import com.amazon.randomcutforest.parkservices.calibration.ErrorHandler;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.PredictiveRandomCutForestMapper;
import com.amazon.randomcutforest.state.statistics.DeviationMapper;

public class ErrorHandlerMapper implements IStateMapper<ErrorHandler, ErrorHandlerState> {

    @Override
    public ErrorHandlerState toState(ErrorHandler model) {
        ErrorHandlerState errorHandlerState = new ErrorHandlerState();
        errorHandlerState.setSequenceIndex(model.getSequenceIndex());
        errorHandlerState.setPercentile(model.getPercentile());
        errorHandlerState.setForecastHorizon(model.getForecastHorizon());
        errorHandlerState.setErrorHorizon(model.getErrorHorizon());
        errorHandlerState.setLastDataDeviations(model.getLastDataDeviations());
        DeviationMapper deviationMapper = new DeviationMapper();
        errorHandlerState.setDeviationStates(getStates(model.getDeviationList(), deviationMapper));
        errorHandlerState.setLastInput(model.getLastInputs());
        errorHandlerState.setInputLength(model.getInputLength());
        errorHandlerState.setPastForecastsFlattened(model.getPastForecastsFlattened());
        if (model.getEstimator() != null) {
            PredictiveRandomCutForestMapper mapper = new PredictiveRandomCutForestMapper();
            errorHandlerState.setEstimatorState(mapper.toState(model.getEstimator()));
        }
        errorHandlerState.setLowerLimit(model.getLowerLimit());
        errorHandlerState.setUpperLimit(model.getUpperLimit());
        return errorHandlerState;
    }

    @Override
    public ErrorHandler toModel(ErrorHandlerState state, long seed) {
        PredictiveRandomCutForest forest = null;
        PredictiveRandomCutForestMapper mapper = new PredictiveRandomCutForestMapper();
        if (state.getEstimatorState() != null) {
            forest = mapper.toModel(state.getEstimatorState());
        }
        DeviationMapper deviationMapper = new DeviationMapper();
        ErrorHandler errorHandler = new ErrorHandler(state.getErrorHorizon(), state.getForecastHorizon(),
                state.getSequenceIndex(), state.getPercentile(), state.getInputLength(),
                state.getPastForecastsFlattened(), state.getLastDataDeviations(), state.getLastInput(),
                getDeviations(state.getDeviationStates(), deviationMapper), forest, null);
        errorHandler.setUpperLimit(state.getUpperLimit());
        errorHandler.setLowerLimit(state.getLowerLimit());
        return errorHandler;
    }
}
