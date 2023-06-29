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

package com.amazon.randomcutforest.parkservices.state.predictorcorrector;

import com.amazon.randomcutforest.parkservices.state.statistics.DeviationState;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderState;
import lombok.Data;

import java.io.Serializable;

import static com.amazon.randomcutforest.state.Version.V3_8;

@Data
public class PredictorCorrectorState implements Serializable {
    private static final long serialVersionUID = 1L;

    private String version = V3_8;
    private BasicThresholderState[] thresholderStates;
    private double[] lastScore;
    private String lastStrategy;
    private int numberOfAttributors;
    private int baseDimension;
    private long randomSeed;
    private double noiseFactor;
    private boolean autoAdjust;
    private double[] modeInformation; // multiple modes -- to be used in future
    private DeviationState[] deviationStates; // in future to be used for learning deviations
    private double[] ignoreNearExpected;

}
