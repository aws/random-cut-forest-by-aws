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

import static com.amazon.randomcutforest.state.Version.V4_0;

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.state.statistics.DeviationState;

@Data
public class PreprocessorState implements Serializable {
    private static final long serialVersionUID = 1L;

    private String version = V4_0;
    private double useImputedFraction;
    private String imputationMethod;
    private String forestMode;

    private String transformMethod;
    private double[] weights;
    private double[] lastShingledPoint;
    private double[] lastShingledInput;
    private double[] defaultFill;
    private double timeDecay;
    private int startNormalization;
    private int stopNormalization;
    private int shingleSize;
    private int dimensions;
    private int inputLength;
    private double clipFactor;
    private boolean normalizeTime;
    private long[] initialTimeStamps;
    private double[][] initialValues;
    private long[] previousTimeStamps;
    private int valuesSeen;
    private int internalTimeStamp;
    @Deprecated
    private DeviationState dataQualityState;
    @Deprecated
    private DeviationState timeStampDeviationState;
    private DeviationState[] deviationStates;

    private DeviationState[] dataQualityStates;
    private DeviationState[] timeStampDeviationStates;
    private boolean fastForward;
}
