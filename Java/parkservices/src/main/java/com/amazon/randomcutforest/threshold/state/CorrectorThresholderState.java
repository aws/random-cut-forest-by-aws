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

package com.amazon.randomcutforest.threshold.state;

import static com.amazon.randomcutforest.threshold.state.Version.V2_1;

import lombok.Data;

import com.amazon.randomcutforest.returntypes.DiVector;

@Data
public class CorrectorThresholderState {

    private String version = V2_1;

    private long randomseed;

    private boolean inAnomaly;

    private double elasticity;

    private boolean attributionEnabled;

    private int count;

    private double discount;

    private int baseDimension;

    private int shingleSize;

    private int minimumScores;

    private DeviationState simpleDeviationState;

    private int lastAnomalyTimeStamp;

    private double lastAnomalyScore;

    private DiVector lastAnomalyAttribution;

    private DeviationState scoreDiffState;

    private double lastScore;

    private boolean ignoreSimilar;

    private boolean previousIsPotentialAnomaly;

    private double absoluteScoreFraction;

    private double upperThreshold;

    private double lowerThreshold;

    private double initialThreshold;

    private double zFactor;

    private double triggerFactor;

    private double upperZfactor;

    private double ignoreSimilarFactor;
}
