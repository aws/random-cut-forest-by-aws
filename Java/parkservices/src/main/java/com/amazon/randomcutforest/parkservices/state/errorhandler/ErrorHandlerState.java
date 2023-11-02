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

import static com.amazon.randomcutforest.state.Version.V4_0;

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.state.PredictiveRandomCutForestState;
import com.amazon.randomcutforest.state.statistics.DeviationState;

@Data
public class ErrorHandlerState implements Serializable {
    private static final long serialVersionUID = 1L;
    private String version = V4_0;
    private int sequenceIndex;
    private double percentile;
    private int forecastHorizon;
    private int errorHorizon;
    private float[] pastForecastsFlattened;
    private int inputLength;
    private float[] lastDataDeviations;
    private double[] lastInput;

    private float[] upperLimit;
    private float[] lowerLimit;
    private DeviationState[] deviationStates;
    private PredictiveRandomCutForestState estimatorState;
    // items below are not used now. Kept for regret computation later.
    // Regret is what we feel when we realize that we should have been better off
    // had we done something else. A basic requirement of regret computation is that
    // it should avoid or at least reduce the regret that will be felt.
    private float[] addersFlattened;
    private float[] multipliersFlattened;
}
