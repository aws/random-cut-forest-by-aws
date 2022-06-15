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

    @Override
    public double[] invert(double[] values, double[] correspondingInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = (weights[i] == 0) ? 0 : values[i] / weights[i];
        }
        return output;
    }

    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput) {
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

    @Override
    public void updateDeviation(double[] inputPoint, double[] lastInput) {
        checkArgument(inputPoint.length * 2 == deviations.length, "incorrect lengths");
        for (int i = 0; i < inputPoint.length; i++) {
            deviations[i].update(inputPoint[i]);
            if (deviations[i + inputPoint.length].getCount() == 0) {
                deviations[i + inputPoint.length].update(0);
            } else {
                deviations[i + inputPoint.length].update(inputPoint[i] - lastInput[i]);
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
