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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class DifferenceTransformer extends WeightedTransformer {

    public DifferenceTransformer(double[] weights, Deviation[] deviation) {
        super(weights, deviation);
    }

    public double[] invert(double[] values, double[] correspondingInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double newValue = correspondingInput[i] + values[i];
            output[i] = (weights[i] == 0) ? 0 : newValue / weights[i];
        }
        return output;
    }

    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput) {
        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;
        double[] last = Arrays.copyOf(lastInput, lastInput.length);
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                ranges.shift(i * baseDimension + j, (float) last[j]);
                float weight = (weights[j] == 0) ? 0f : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
                last[j] = ranges.values[j];
            }
        }
    }

    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] lastInput, double[] factors,
            double clipFactor) {

        double[] input = new double[inputPoint.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (internalTimeStamp == 0) ? 0 : weights[i] * (inputPoint[i] - lastInput[i]);
        }
        return input;
    }

}
