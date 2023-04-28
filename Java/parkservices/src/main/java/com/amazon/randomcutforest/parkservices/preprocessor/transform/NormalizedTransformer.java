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

package com.amazon.randomcutforest.parkservices.preprocessor.transform;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class NormalizedTransformer extends WeightedTransformer {

    public NormalizedTransformer(double[] weights, Deviation[] deviation) {
        super(weights, deviation);
    }

    protected double clipValue(double clipfactor) {
        return clipfactor;
    }

    protected double getScale(int i, Deviation[] devs) {
        return (Math.abs(devs[i + 2 * weights.length].getMean()) + 1.0);
    }

    protected double getShift(int i, Deviation[] devs) {
        return devs[i].getMean();
    }

}
