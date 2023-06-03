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

import static com.amazon.randomcutforest.state.Version.V3_8;

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorState;
import com.amazon.randomcutforest.parkservices.state.preprocessor.PreprocessorState;
import com.amazon.randomcutforest.parkservices.state.returntypes.ComputeDescriptorState;
import com.amazon.randomcutforest.parkservices.state.threshold.BasicThresholderState;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.state.returntypes.DiVectorState;

@Data
public class ThresholdedRandomCutForestState implements Serializable {
    private static final long serialVersionUID = 1L;

    private String version = V3_8;
    RandomCutForestState forestState;
    // deprecated but not marked due to 2.1 models
    private BasicThresholderState thresholderState;
    private PreprocessorState[] preprocessorStates;

    // following fields are deprecated, but not removed for compatibility with 2.1
    // moels
    private double ignoreSimilarFactor;
    private double triggerFactor;
    private long lastAnomalyTimeStamp;
    private double lastAnomalyScore;
    private DiVectorState lastAnomalyAttribution;
    private double lastScore;
    private double[] lastAnomalyPoint;
    private double[] lastExpectedPoint;
    private boolean previousIsPotentialAnomaly;
    private boolean inHighScoreRegion;
    private boolean ignoreSimilar;
    private int numberOfAttributors;
    // end deprecated segment

    private long randomSeed;

    private String forestMode;
    private String transformMethod;
    private String scoringStrategy;
    @Deprecated
    private int lastRelativeIndex;
    private int lastReset;
    private PredictorCorrectorState predictorCorrectorState;
    private ComputeDescriptorState lastDescriptorState;

}
