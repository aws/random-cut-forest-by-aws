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

    double[] getCurrentInput();

    double[] getExpectedRCFPoint();

    double[] getRCFPoint();

    int getRelativeIndex();

    double getRCFScore();

    DiVector getAttribution();

    long getInternalTimeStamp();

    ForestMode getForestMode();

    TransformMethod getTransformMethod();

    RCFComputeDescriptor copyOf();

}
