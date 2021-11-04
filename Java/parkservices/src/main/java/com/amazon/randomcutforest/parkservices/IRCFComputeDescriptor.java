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

package com.amazon.randomcutforest.parkservices;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.returntypes.DiVector;

public interface IRCFComputeDescriptor {

    // the current input point; can have missing values
    double[] getCurrentInput();

    // potential missing values in the current input (ideally null)
    int[] getMissingValues();

    // the point which the RCF "expects" -- corresponds to low likelihood
    double[] getExpectedRCFPoint();

    // the point corresponding to the currentInput which is used in RCF for scoring
    double[] getRCFPoint();

    // time slice of the largest contributor to the score (non-positive and relative
    // to current
    // point (RCFPoint) between [-shingleSize+1,0])
    int getRelativeIndex();

    // the score on RCFPoint
    double getRCFScore();

    // the attribution of the entire shingled RCFPoint
    DiVector getAttribution();

    // the timestamp corresponding to RCFPoint (can be more than the number of
    // updates with imputation)
    long getInternalTimeStamp();

    // forestMode
    ForestMode getForestMode();

    // transformation method (if used)
    TransformMethod getTransformMethod();

    // an explicit copy operator
    RCFComputeDescriptor copyOf();

}
