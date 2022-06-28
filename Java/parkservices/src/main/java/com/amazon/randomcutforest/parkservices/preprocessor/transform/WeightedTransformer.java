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

/**
 * A weighted transformer maintains 2X data structures that measure discounted
 * averages and the corresponding standard deviation. The element i corresponds
 * to discounted average of the variable i and element (X+i) corresponds to the
 * discounted average of the single step differences of the same variable i.
 * These two quantities together can help answer a number of estimation
 * questions of a time series, and in particular help solve for simple linear
 * drifts. Even though the discounted averages are not obviously required --
 * they are useful in forecasts.
 *
 * We note that more complicated drifts may require different (and complicated)
 * solutions in a streaming context and are not implemented yet.
 *
 * It transforms the variable i by multiplying with weight[i].
 *
 *
 */
@Getter
@Setter
public class WeightedTransformer implements ITransformer {

    double[] weights;

    Deviation[] deviations;

    public WeightedTransformer(double[] weights, Deviation[] deviations) {
        checkArgument(2 * weights.length == deviations.length, "incorrect lengths");
        this.weights = Arrays.copyOf(weights, weights.length);
        this.deviations = new Deviation[deviations.length];
        for (int i = 0; i < deviations.length; i++) {
            this.deviations[i] = deviations[i].copy();
        }
    }

    /**
     * the inversion does not require previousInput; note that weight == 0, would
     * produce 0 values in the inversion
     *
     * @param values        what the RCF would like to observe
     * @param previousInput what was the real (or previously imputed) observation
     * @return the observations that would (approximately) transform to the array
     *         values[]
     */
    public double[] invert(double[] values, double[] previousInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = (weights[i] == 0) ? 0 : values[i] / weights[i];
        }
        return output;
    }

    /**
     * inverts a forecast (and upper and lower limits) provided by RangeVector range
     * note that the expected difference maintained in deviation[j + inputLength] is
     * added for each attribute j
     * 
     * @param ranges        provides p50 values with upper and lower estimates
     * @param baseDimension the number of variables being forecast (often 1)
     * @param previousInput the last input of length baseDimension
     */
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput) {
        int horizon = ranges.values.length / baseDimension;
        int inputLength = weights.length;
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                float weight = (weights[j] == 0) ? 0 : 1.0f / (float) weights[j];
                ranges.shift(i * baseDimension + j, (float) ((i + 1) * deviations[j + inputLength].getMean()));
                ranges.scale(i * baseDimension + j, weight);
            }
        }
    }

    /**
     * updates the 2*inputPoint.length statistics; the statistic i corresponds to
     * discounted average of variable i and statistic i + inputPoint.length
     * corresponds to the discounted average single step difference
     * 
     * @param inputPoint    the input seen by TRCF
     * @param previousInput the previous input
     */
    public void updateDeviation(double[] inputPoint, double[] previousInput) {
        checkArgument(inputPoint.length * 2 == deviations.length, "incorrect lengths");
        for (int i = 0; i < inputPoint.length; i++) {
            deviations[i].update(inputPoint[i]);
            if (deviations[i + inputPoint.length].getCount() == 0) {
                deviations[i + inputPoint.length].update(0);
            } else {
                deviations[i + inputPoint.length].update(inputPoint[i] - previousInput[i]);
            }
        }
    }

    /**
     * a transformation that transforms the multivariate values by multiplying with
     * weights
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
        for (int i = 0; i < inputPoint.length; i++) {
            input[i] = inputPoint[i] * weights[i];
        }
        return input;
    }

    public Deviation[] getDeviations() {
        Deviation[] answer = new Deviation[deviations.length];
        for (int i = 0; i < deviations.length; i++) {
            answer[i] = deviations[i].copy();
        }
        return answer;
    }

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public void setWeights(double[] weights) {
        checkArgument(weights.length == this.weights.length, " incorrect length");
        this.weights = Arrays.copyOf(weights, weights.length);
    }
}
