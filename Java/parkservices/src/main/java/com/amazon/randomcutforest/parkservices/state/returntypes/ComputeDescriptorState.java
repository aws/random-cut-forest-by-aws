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

package com.amazon.randomcutforest.parkservices.state.returntypes;

import com.amazon.randomcutforest.state.returntypes.DiVectorState;
import lombok.Data;

import java.io.Serializable;

@Data
public class ComputeDescriptorState implements Serializable {
    private static final long serialVersionUID = 1L;

    private long lastAnomalyTimeStamp;
    private double lastAnomalyScore;
    private DiVectorState lastAnomalyAttribution;
    private double lastScore;
    private double[] lastAnomalyPoint;
    private double[] lastExpectedPoint;
    private int lastRelativeIndex;
    private int lastReset;
    private String lastStrategy;
    private double[] lastShift;
    private double[] lastScale;
    private double[] lastPostShift;
    private double transformDecay;
    private double[] postDeviations;
}
