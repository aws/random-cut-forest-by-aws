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

    @Override
    public double[] invert(double[] values, double[] previousInput) {
        double[] output = super.invert(values, previousInput);
        for (int i = 0; i < values.length; i++) {
            output[i] += previousInput[i];
        }
        return output;
    }

    /**
     * inverts a forecast (and upper and lower limits) provided by RangeVector range
     * the values are scaled by the factor used in the transformation for each
     * iteration; and the resulting value is added back as an inverse of the
     * differencing operation.
     * 
     * @param ranges        provides p50 values with upper and lower estimates
     * @param baseDimension the number of variables being forecast (often 1)
     * @param previousInput the last input of length baseDimension
     */
    @Override
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput) {
        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;
        double[] last = Arrays.copyOf(previousInput, previousInput.length);
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                float weight = (weights[j] == 0) ? 0f : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
                ranges.shift(i * baseDimension + j, (float) last[j]);
                last[j] = ranges.values[j];
            }
        }
    }

    @Override
    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] previousInput,
            Deviation[] initials, double clipFactor) {

        double[] input = new double[inputPoint.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (internalTimeStamp == 0) ? 0 : (inputPoint[i] - previousInput[i]);
        }
        return super.transformValues(internalTimeStamp, input, null, initials, clipFactor);
    }

}
