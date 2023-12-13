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

package com.amazon.randomcutforest.preprocessor.transform;

import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.statistics.Deviation;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SubtractMATransformer extends WeightedTransformer {

    public SubtractMATransformer(double[] weights, Deviation[] deviations) {
        super(weights, deviations);
    }

    @Override
    protected double getShift(int i, Deviation[] devs) {
        return devs[i].getMean();
    }

    @Override
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput,
                                    double[] correction) {
        super.invertForecastRange(ranges,baseDimension,previousInput,correction);
        int horizon = ranges.values.length / baseDimension;
        int inputLength = weights.length;
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                ranges.shift(i * baseDimension + j,
                        (float) (0.5 * getDrift(j, deviations)));
            }
        }
    }
}
