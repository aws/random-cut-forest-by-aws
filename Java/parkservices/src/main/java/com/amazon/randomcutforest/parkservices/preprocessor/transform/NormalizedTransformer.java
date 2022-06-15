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
public class NormalizedTransformer extends WeightedTransformer {

    // in case of normalization, uses this constant in denominator to ensure
    // smoothness near 0
    public static double DEFAULT_NORMALIZATION_PRECISION = 1e-3;

    public NormalizedTransformer(double[] weights, Deviation[] deviation) {
        super(weights, deviation);
    }

    public double[] invert(double[] values, double[] correspondingInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double newValue = deviations[i].getMean()
                    + 2 * values[i] * (deviations[i].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
            output[i] = (weights[i] == 0) ? 0 : newValue / weights[i];
        }
        return output;
    }

    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput) {
        int inputLength = weights.length;
        int horizon = ranges.values.length / baseDimension;
        for (int i = 0; i < horizon; i++) {
            for (int j = 0; j < inputLength; j++) {
                double factor = 2 * (deviations[j].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
                ranges.scale(i * baseDimension + j, (float) factor);
                double shift = deviations[j].getMean() + i * deviations[j + inputLength].getMean();
                ranges.shift(i * baseDimension + j, (float) shift);
                float weight = (weights[j] == 0) ? 0 : 1.0f / (float) weights[j];
                ranges.scale(i * baseDimension + j, weight);
            }
        }
    }

    protected double normalize(double value, Deviation deviation, double factor, double clipFactor) {
        double currentFactor = (factor != 0) ? factor : deviation.getDeviation();
        if (value - deviation.getMean() >= 2 * clipFactor * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return clipFactor;
        }
        if (value - deviation.getMean() < -2 * clipFactor * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return -clipFactor;
        } else {
            // deviation cannot be 0
            return (value - deviation.getMean()) / (2 * (currentFactor + DEFAULT_NORMALIZATION_PRECISION));
        }
    }

    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] lastInput, double[] factors,
            double clipFactor) {
        double[] output = new double[inputPoint.length];
        for (int i = 0; i < inputPoint.length; i++) {
            output[i] = weights[i]
                    * normalize(inputPoint[i], deviations[i], (factors == null) ? 0 : factors[i], clipFactor);
        }
        return output;
    }

}
