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

    /**
     * note that weight == 0, would produce 0 values in the inversion
     *
     * @param values        what the RCF would like to observe
     * @param previousInput what was the real (or previously imputed) observation
     * @return the observations that would (approximately) transform to values[]
     */
    public double[] invert(double[] values, double[] previousInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double newValue = previousInput[i] + values[i];
            output[i] = (weights[i] == 0) ? 0 : newValue / weights[i];
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
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput) {
        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;
        double[] last = Arrays.copyOf(previousInput, previousInput.length);
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                ranges.shift(i * baseDimension + j, (float) last[j]);
                float weight = (weights[j] == 0) ? 0f : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
                last[j] = ranges.values[j];
            }
        }
    }

    /**
     * a transformation that differences the multivariate values
     * 
     * @param internalTimeStamp timestamp corresponding to this operation; used to
     *                          ensure smoothness at 0
     * @param inputPoint        the actual input
     * @param previousInput     the previous input
     * @param factors           an array containing normalization factors, used only
     *                          for the initial segment; otherwise it is null
     * @param clipFactor        the factor used in clipping the normalized values
     * @return the transformed values to be shingled and used in RCF
     */
    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] previousInput,
            double[] factors, double clipFactor) {

        double[] input = new double[inputPoint.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (internalTimeStamp == 0) ? 0 : weights[i] * (inputPoint[i] - previousInput[i]);
        }
        return input;
    }

}
