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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class NormalizedDifferenceTransformer extends NormalizedTransformer {

    public NormalizedDifferenceTransformer(double[] weights, Deviation[] deviation) {
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
     * the values are scaled by the factor used in the transformation note that the
     * expected difference maintained in deviation[j + inputLength] is added for
     * each attribute j, once for each iteration; and the resulting value is added
     * back as an inverse of the differencing operation.
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
                double weight = (weights[j] == 0) ? 0 : getScale(j, deviations) / weights[j];
                ranges.scale(i * baseDimension + j, (float) weight);
                double shift = last[j] + getShift(j, deviations);
                ranges.shift(i * baseDimension + j, (float) shift);
                last[j] = ranges.values[i * baseDimension + j];
            }
        }
    }

    /**
     * a transformation that differences and then normalizes the results of
     * multivariate values
     * 
     * @param internalTimeStamp timestamp corresponding to this operation; used to
     *                          ensure smoothness at 0
     * @param inputPoint        the actual input
     * @param previousInput     the previous input
     * @param initials          an array containing normalization statistics, used
     *                          only for the initial segment; otherwise it is null
     * @param clipFactor        the factor used in clipping the normalized values
     * @return the transformed values to be shingled and used in RCF
     */
    @Override
    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] previousInput,
            Deviation[] initials, double clipFactor) {
        double[] input = new double[inputPoint.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (internalTimeStamp == 0) ? 0 : inputPoint[i] - previousInput[i];
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

    @Override
    protected double getShift(int i, Deviation[] devs) {
        return devs[i + weights.length].getMean();
    }

    @Override
    protected double getScale(int i, Deviation[] devs) {
        return (devs[i + weights.length].getDeviation() + 1.0);
    }

}
