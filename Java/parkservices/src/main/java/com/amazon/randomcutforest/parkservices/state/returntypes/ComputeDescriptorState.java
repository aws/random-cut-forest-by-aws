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

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.state.returntypes.DiVectorState;

@Data
public class ComputeDescriptorState implements Serializable {
    private static final long serialVersionUID = 1L;

    private long internalTimeStamp;
    private double score;
    private DiVectorState attribution;
    private double lastScore;
    private double[] point;
    private double[] expectedPoint;
    private int relativeIndex;
    private int lastReset;
    private String strategy;
    private double[] shift;
    private double[] scale;
    private double[] postShift;
    private double transformDecay;
    private double[] postDeviations;
    private double threshold;
    private double anomalyGrade;
    private String correctionMode;
}
