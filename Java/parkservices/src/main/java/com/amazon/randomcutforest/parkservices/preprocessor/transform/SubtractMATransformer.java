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
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class SubtractMATransformer extends WeightedTransformer {

    public SubtractMATransformer(double[] weights, Deviation[] deviations) {
        super(weights, deviations);
    }

    @Override
    public double[] invert(double[] values, double[] correspondingInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = (weights[i] == 0) ? 0 : (values[i] + deviations[i].getMean()) / weights[i];
        }
        return output;
    }

    @Override
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput) {

        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;

        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                // both of these are lagging averages -- that can be an issue
                double shift = deviations[j].getMean() + (i + 1) * deviations[j + inputLength].getMean();
                ranges.shift(i * baseDimension + j, (float) shift);
                float weight = (weights[j] == 0) ? 0f : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
            }
        }
    }

    /**
     * transforms the values based on transformMethod
     * 
     * @param inputPoint the actual input
     * @param factors    an array containing normalization factors, used only for
     *                   the initial segment; otherwise it is null
     * @return the transformed input
     */
    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] lastInput, double[] factors,
            double clipFactor) {
        double[] output = new double[inputPoint.length];
        for (int i = 0; i < inputPoint.length; i++) {
            output[i] = (internalTimeStamp == 0) ? 0 : weights[i] * (inputPoint[i] - deviations[i].getMean());
        }
        return output;
    }

}
