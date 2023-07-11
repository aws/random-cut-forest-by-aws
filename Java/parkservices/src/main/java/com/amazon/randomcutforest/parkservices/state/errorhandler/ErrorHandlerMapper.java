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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Locale;

import com.amazon.randomcutforest.parkservices.ErrorHandler;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.state.IStateMapper;

public class ErrorHandlerMapper implements IStateMapper<ErrorHandler, ErrorHandlerState> {

    @Override
    public ErrorHandlerState toState(ErrorHandler model) {
        ErrorHandlerState errorHandlerState = new ErrorHandlerState();
        errorHandlerState.setSequenceIndex(model.getSequenceIndex());
        errorHandlerState.setPercentile(model.getPercentile());
        errorHandlerState.setForecastHorizon(model.getForecastHorizon());
        errorHandlerState.setErrorHorizon(model.getErrorHorizon());
        errorHandlerState.setLastDeviations(model.getLastDeviations());

        // pastForecasts[i] contains forecasts at timestamp i. We have three float
        // arrays:
        // upper, lower, values. Each array is of length forecastHorizon*dimensions
        // since
        // we have forecastHorizon forecasts per dimension.
        RangeVector[] pastForecasts = model.getPastForecasts();
        float[][] actuals = model.getActuals();
        int arrayLength = pastForecasts.length;
        checkArgument(pastForecasts != null, "pastForecasts cannot be null");
        checkArgument(actuals != null, "actuals cannot be null");
        checkArgument(arrayLength == actuals.length, String.format(Locale.ROOT,
                "actuals array length %d and pastForecasts array length %d is not equal", actuals.length, arrayLength));
        int forecastHorizon = model.getForecastHorizon();
        float[] pastForecastsFlattened = null;
        int inputLength = 0;
        if (pastForecasts.length == 0 || pastForecasts[0].values == null || pastForecasts[0].values.length == 0) {
            pastForecastsFlattened = new float[0];
        } else {
            int pastForecastsLength = pastForecasts[0].values.length;
            inputLength = pastForecastsLength / forecastHorizon;
            pastForecastsFlattened = new float[arrayLength * 3 * forecastHorizon * inputLength];

            for (int i = 0; i < arrayLength; i++) {
                System.arraycopy(pastForecasts[i].values, 0, pastForecastsFlattened, 3 * i * pastForecastsLength,
                        pastForecastsLength);
                System.arraycopy(pastForecasts[i].upper, 0, pastForecastsFlattened, (3 * i + 1) * pastForecastsLength,
                        pastForecastsLength);
                System.arraycopy(pastForecasts[i].lower, 0, pastForecastsFlattened, (3 * i + 2) * pastForecastsLength,
                        pastForecastsLength);
            }
        }
        errorHandlerState.setInputLength(inputLength);
        errorHandlerState.setPastForecastsFlattened(pastForecastsFlattened);

        float[] actualsFlattened = null;
        if (actuals.length == 0 || actuals[0].length == 0) {
            actualsFlattened = new float[0];
        } else {
            actualsFlattened = new float[arrayLength * inputLength];
            for (int i = 0; i < arrayLength; i++) {
                System.arraycopy(actuals[i], 0, actualsFlattened, i * inputLength, inputLength);
            }
        }
        errorHandlerState.setActualsFlattened(actualsFlattened);
        return errorHandlerState;
    }

    @Override
    public ErrorHandler toModel(ErrorHandlerState state, long seed) {
        return new ErrorHandler(state.getErrorHorizon(), state.getForecastHorizon(), state.getSequenceIndex(),
                state.getPercentile(), state.getInputLength(), state.getActualsFlattened(),
                state.getPastForecastsFlattened(), state.getLastDeviations(), null);
    }

}
