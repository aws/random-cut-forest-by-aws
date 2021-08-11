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

package com.amazon.randomcutforest.ERCF;

import static com.amazon.randomcutforest.ERCF.Version.V2_1;

import lombok.Data;

import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.threshold.state.CorrectorThresholderState;

@Data
public class ERCFState {
    private final String version = V2_1;
    private RandomCutForestState forestState;
    private CorrectorThresholderState thresholderState;
    private int count;
    private int numberOfAttributors;
    private double[] lastAnomalyPoint;
    private int lastAnomalyTimeStamp;
}
