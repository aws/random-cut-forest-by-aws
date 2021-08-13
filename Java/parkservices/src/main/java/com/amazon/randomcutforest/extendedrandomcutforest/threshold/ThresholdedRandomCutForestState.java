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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold;

import com.amazon.randomcutforest.extendedrandomcutforest.threshold.state.BasicThresholderState;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.state.RandomCutForestState;
import lombok.Data;

import static com.amazon.randomcutforest.extendedrandomcutforest.threshold.state.Version.V2_1;

@Data
public class ThresholdedRandomCutForestState {
    private String version = V2_1;
    RandomCutForestState forestState;
    private BasicThresholderState thresholderState;
    private double ignoreSimilarFactor;
    private double triggerFactor;
    private long lastAnomalyTimeStamp;
    private double lastAnomalyScore;
    private DiVector lastAnomalyAttribution;
    private double lastScore;
    private double[] lastAnomalyPoint;
    private double[] lastExpectedPoint;
    private boolean previousIsPotentialAnomaly;
    private boolean inAnomaly;
    private boolean ignoreSimilar;
    private int numberOfAttributors;
    private long randomseed;
}
