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

    @Deprecated
    private boolean inAnomaly;

    @Deprecated
    private double elasticity;

    @Deprecated
    private boolean attributionEnabled;

    private int count;

    private int minimumScores;

    // do not use
    private DeviationState primaryDeviationState;

    // do not use
    private DeviationState secondaryDeviationState;

    // do not use
    private DeviationState thresholdDeviationState;

    @Deprecated
    private double upperThreshold;

    private double lowerThreshold;

    private double absoluteThreshold;

    private boolean autoThreshold;

    private double initialThreshold;

    private double zFactor;

    @Deprecated
    private double upperZfactor;

    @Deprecated
    private double absoluteScoreFraction;

    private double horizon;

    private DeviationState[] deviationStates;

}
