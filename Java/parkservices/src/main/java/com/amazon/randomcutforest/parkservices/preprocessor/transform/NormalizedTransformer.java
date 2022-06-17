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

    /**
     * the inversion does not require previousInput; note that weight == 0, would
     * produce 0 values in the inversion
     *
     * @param values        what the RCF would like to observe
     * @param previousInput what was the real (or previously imputed) observation
     * @return the observations that would (approximately) transform to values[]
     */
    public double[] invert(double[] values, double[] previousInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double newValue = deviations[i].getMean()
                    + 2 * values[i] * (deviations[i].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
            output[i] = (weights[i] == 0) ? 0 : newValue / weights[i];
        }
        return output;
    }

    /**
     * inverts a forecast (and upper and lower limits) provided by RangeVector range
     * the values are scaled by the factor used in normalization note that the
     * expected difference maintained in deviation[j + inpulLength] is added for
     * each attribute j, along with the mean from the normalization
     * 
     * @param ranges        provides p50 values with upper and lower estimates
     * @param baseDimension the number of variables being forecast (often 1)
     * @param previousInput the last input of length baseDimension
     */
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput) {
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

    /**
     * a normalization function
     * 
     * @param value      argument to be normalized
     * @param deviation  a statistic that estimates discounted average and the
     *                   discounted standard deviation
     * @param factor     an alternate set of factors that are used at
     *                   initialization; the specific use case corresponds to
     *                   bufferring 10 or 20 values and then starting the
     *                   normalization; otherwise the initial values would have
     *                   oversize impact
     * @param clipFactor the output value is bound is in [-clipFactor,clipFactor]
     * @return the normalized value
     */
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

    /**
     * a transformation that normalizes the multivariate values
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
        double[] output = new double[inputPoint.length];
        for (int i = 0; i < inputPoint.length; i++) {
            output[i] = weights[i]
                    * normalize(inputPoint[i], deviations[i], (factors == null) ? 0 : factors[i], clipFactor);
        }
        return output;
    }

}
