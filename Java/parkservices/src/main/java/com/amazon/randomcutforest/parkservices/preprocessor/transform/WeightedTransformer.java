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
 * A weighted transformer maintains several data structures ( currently 3X) that
 * measure discounted averages and the corresponding standard deviations. for
 * input length X. The element i corresponds to discounted average of the
 * variable i, element (X+i) corresponds to the discounted average of the single
 * step differences of the same variable i, and element (2X+i) corresponds to
 * difference of variable i and the dicounted average, to capture second order
 * differences These quantities together can help answer a number of estimation
 * questions of a time series, and in particular help solve for simple linear
 * drifts. Even though the discounted averages are not obviously required --
 * they are useful in forecasts.
 *
 */
@Getter
@Setter
public class WeightedTransformer implements ITransformer {

    public static int NUMBER_OF_STATS = 3;

    double[] weights;

    Deviation[] deviations;

    public WeightedTransformer(double[] weights, Deviation[] deviations) {
        checkArgument(NUMBER_OF_STATS * weights.length == deviations.length, "incorrect lengths");
        this.weights = Arrays.copyOf(weights, weights.length);
        this.deviations = new Deviation[deviations.length];
        for (int i = 0; i < deviations.length; i++) {
            checkArgument(deviations[i] != null, "cannot be null");
            this.deviations[i] = deviations[i].copy();
        }
    }

    /**
     * the inversion does not require previousInput; note that weight == 0, would
     * produce 0 values in the inversion
     *
     * @param values        what the RCF would like to observe
     * @param previousInput what was the real (or previously imputed) observation
     * @return the observations that would (approximately) transform to values[]
     */
    @Override
    public double[] invert(double[] values, double[] previousInput) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = (weights[i] == 0) ? 0 : values[i] * getScale(i, deviations) / weights[i];
            output[i] += getShift(i, deviations);
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
                double weight = (weights[j] == 0) ? 0 : getScale(j, deviations) / weights[j];
                ranges.scale(i * baseDimension + j, (float) weight);
                ranges.shift(i * baseDimension + j,
                        (float) (getShift(j, deviations) + (i + 1) * deviations[j + inputLength].getMean()));
            }
        }
    }

    /**
     * updates the 3*inputPoint.length statistics; the statistic i corresponds to
     * discounted average of variable i and statistic i + inputPoint.length
     * corresponds to the discounted average single step difference
     * 
     * @param inputPoint    the input seen by TRCF
     * @param previousInput the previous input
     */
    public void updateDeviation(double[] inputPoint, double[] previousInput) {
        checkArgument(inputPoint.length * NUMBER_OF_STATS == deviations.length, "incorrect lengths");
        checkArgument(inputPoint.length == previousInput.length, " lengths must match");
        for (int i = 0; i < inputPoint.length; i++) {
            deviations[i].update(inputPoint[i]);
            if (deviations[i + inputPoint.length].getCount() == 0) {
                deviations[i + inputPoint.length].update(0);
            } else {
                deviations[i + inputPoint.length].update(inputPoint[i] - previousInput[i]);
            }
            deviations[i + 2 * inputPoint.length].update(deviations[i].getDeviation());
        }
    }

    /**
     * a normalization function
     *
     * @param value      argument to be normalized
     * @param shift      the shift in the value
     * @param scale      the scaling factor
     * @param clipFactor the output value is bound is in [-clipFactor,clipFactor]
     * @return the normalized value
     */
    protected double normalize(double value, double shift, double scale, double clipFactor) {
        checkArgument(scale > 0, " should be non-negative");
        double t = (value - shift) / (scale);
        if (t >= clipFactor) {
            return clipFactor;
        }
        if (t < -clipFactor) {
            return -clipFactor;
        }
        return t;
    }

    /**
     * a transformation that normalizes the multivariate values
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
        double[] output = new double[inputPoint.length];
        for (int i = 0; i < inputPoint.length; i++) {
            Deviation[] devs = (initials == null) ? deviations : initials;
            output[i] = weights[i]
                    * normalize(inputPoint[i], getShift(i, devs), getScale(i, devs), clipValue(clipFactor));
        }
        return output;
    }

    protected double clipValue(double clipfactor) {
        return Double.MAX_VALUE;
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

    protected double getScale(int i, Deviation[] devs) {
        return (1.0);
    }

    protected double getShift(int i, Deviation[] devs) {
        return 0;
    }

    @Override
    public double[] getScale() {
        double[] answer = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            answer[i] = (weights[i] == 0) ? 0 : getScale(i, deviations) / weights[i];
        }
        return answer;
    }

    @Override
    public double[] getShift(double[] previous) {
        double[] answer = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            answer[i] = getShift(i, deviations);
        }
        return answer;
    }

    public double[] getSmoothedDeviations() {
        checkArgument(deviations.length >= 3 * weights.length, "incorrect call");
        double[] answer = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            answer[i] = Math.abs(deviations[i + 2 * weights.length].getMean());
        }
        return answer;
    }
}
