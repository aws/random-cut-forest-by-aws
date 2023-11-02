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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.statistics.Deviation;

@Getter
@Setter
public class DifferenceTransformer extends WeightedTransformer {

    public DifferenceTransformer(double[] weights, Deviation[] deviation) {
        super(weights, deviation);
    }

    @Override
    public void invert(float[] values, double[] previousInput) {
        super.invert(values, previousInput);
        for (int i = 0; i < values.length; i++) {
            double output = values[i] + previousInput[i];
            values[i] = (float) output;
        }
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
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput,
            double[] correction) {
        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;
        double[] last = Arrays.copyOf(previousInput, previousInput.length);
        checkArgument(correction.length >= inputLength, " incorrect length ");
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                float weight = (weights[j] == 0) ? 0f : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
                ranges.shift(i * baseDimension + j, (float) (getShift(j, deviations) + last[j]));
                last[j] = ranges.values[j];
            }
        }
    }

    @Override
    public float[] transformValues(int internalTimeStamp, double[] inputPoint, double[] previousInput,
            Deviation[] initials, double clipFactor) {

        double[] input = new double[inputPoint.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (internalTimeStamp == 0) ? 0 : (inputPoint[i] - previousInput[i]);
        }
        return super.transformValues(internalTimeStamp, input, null, initials, clipFactor);
    }

    @Override
    public double[] getShift(double[] previous) {
        double[] answer = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            answer[i] = getShift(i, deviations) + previous[i];
        }
        return answer;
    }
}
