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

package com.amazon.randomcutforest.parkservices.state.threshold;

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.parkservices.state.statistics.DeviationState;

@Data
public class BasicThresholderState implements Serializable {
    private static final long serialVersionUID = 1L;

    private long randomseed;

    private boolean inAnomaly;

    private double elasticity;

    private boolean attributionEnabled;

    private int count;

    private int minimumScores;

    private DeviationState primaryDeviationState;

    private DeviationState secondaryDeviationState;

    private DeviationState thresholdDeviationState;

    private double upperThreshold;

    private double lowerThreshold;

    private double absoluteThreshold;

    private boolean autoThreshold;

    private double initialThreshold;

    private double zFactor;

    private double upperZfactor;

    private double absoluteScoreFraction;

    private double horizon;

}
