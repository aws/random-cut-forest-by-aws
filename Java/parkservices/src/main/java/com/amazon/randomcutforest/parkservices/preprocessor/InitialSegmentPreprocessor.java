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

package com.amazon.randomcutforest.parkservices.preprocessor;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class InitialSegmentPreprocessor extends Preprocessor {

    public InitialSegmentPreprocessor(Builder<?> builder) {
        super(builder);
        initialValues = new double[startNormalization][];
        initialTimeStamps = new long[startNormalization];
    }

    /**
     * given an input produces a scaled transform to be used in the forest
     *
     * @param input     the actual input seen
     * @param timestamp timestamp of said input
     * @return a scaled/transformed input which can be used in the forest
     */
    protected double[] getScaledInput(double[] input, long timestamp, double[] defaultFactors,
            double defaultTimeFactor) {
        double[] scaledInput = transformValues(input, defaultFactors);
        if (mode == ForestMode.TIME_AUGMENTED) {
            scaledInput = augmentTime(scaledInput, timestamp, defaultTimeFactor);
        }
        return scaledInput;
    }

    /**
     * A core function of the preprocessor. It can augment time values (with
     * normalization) or impute missing values on the fly using the forest.
     *
     * @param inputPoint the actual input
     * @param timestamp  timestamp of the point
     * @param forest     RCF
     * @return a scaled/normalized tuple that can be used for anomaly detection
     */
    public double[] preProcess(double[] inputPoint, long timestamp, RandomCutForest forest) {

        if (valuesSeen < startNormalization) {
            storeInitial(inputPoint, timestamp);
            return null;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        return getScaledInput(inputPoint, timestamp, null, 0);
    }

    /**
     * maps the time back. The returned value is an approximation for
     * relativePosition less than 0 which corresponds to an anomaly in the past.
     * Since the state of the statistic is now changed based on more recent values
     *
     * @param gap              estimated value
     * @param relativePosition how far back in the shingle
     * @return transform of the time value to original input space
     */
    public long inverseMapTime(double gap, int relativePosition) {
        // note this ocrresponds to differencing being always on
        checkArgument(shingleSize + relativePosition >= 0, " error");
        double factor = weights[inputLength];
        if (factor == 0) {
            return 0;
        }
        if (normalizeTime) {
            return (long) Math.floor(previousTimeStamps[shingleSize - 1 + relativePosition]
                    + timeStampDeviation.getMean() + 2 * gap * timeStampDeviation.getDeviation() / factor);
        } else {
            return (long) Math.floor(gap / factor + previousTimeStamps[shingleSize - 1 + relativePosition]
                    + timeStampDeviation.getMean());
        }
    }

    /**
     * if we find an estimated value for input index i, then this function inverts
     * that estimate to indicate (approximately) what that value should have been in
     * the actual input space
     *
     * @param value              estimated value
     * @param index              position in the input vector
     * @param relativeBlockIndex the index of the block in the shingle
     * @return the estimated value whose transform would be the value
     */
    @Override
    public double inverseTransform(double value, int index, int relativeBlockIndex) {
        // note that time does not matter for this case
        if (!requireInitialSegment(false, transformMethod)) {
            return super.inverseTransform(value, index, relativeBlockIndex);
        }
        if (transformMethod == TransformMethod.NORMALIZE) {
            double newValue = deviationList[index].getMean()
                    + 2 * value * (deviationList[index].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
            return (weights[index] == 0) ? 0 : newValue / weights[index];
        }
        checkArgument(transformMethod == TransformMethod.NORMALIZE_DIFFERENCE, "incorrect configuration");
        double[] difference = getShingledInput(shingleSize - 1 + relativeBlockIndex);
        double newValue = difference[index] + deviationList[index].getMean()
                + +2 * value * (deviationList[index].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
        return (weights[index] == 0) ? 0 : newValue / weights[index];
    }

    /**
     * stores initial data for normalization
     *
     * @param inputPoint input data
     * @param timestamp  timestamp
     */
    protected void storeInitial(double[] inputPoint, long timestamp) {
        initialTimeStamps[valuesSeen] = timestamp;
        initialValues[valuesSeen] = Arrays.copyOf(inputPoint, inputPoint.length);
        ++valuesSeen;
    }

    /**
     * a modification for the deviations that use differencing, the normalization
     * requires initial segments
     * 
     * @param inputPoint the input point
     */
    @Override
    void updateDeviation(double[] inputPoint) {
        for (int i = 0; i < inputPoint.length; i++) {
            double value = inputPoint[i];
            if (transformMethod == TransformMethod.DIFFERENCE
                    || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
                value -= lastShingledInput[lastShingledInput.length - inputLength + i];
            }
            deviationList[i].update(value);
        }
    }

    /**
     * augments (potentially normalized) input with time (which is always
     * differenced)
     *
     * @param normalized (potentially normalized) input point
     * @param timestamp  timestamp of current point
     * @param timeFactor a factor used in normalizing time
     * @return a tuple with one exta field
     */
    protected double[] augmentTime(double[] normalized, long timestamp, double timeFactor) {
        double[] scaledInput = new double[normalized.length + 1];
        System.arraycopy(normalized, 0, scaledInput, 0, normalized.length);
        if (valuesSeen <= 1) {
            scaledInput[normalized.length] = 0;
        } else {
            double timeshift = timestamp - previousTimeStamps[shingleSize - 1];
            scaledInput[normalized.length] = weights[inputLength]
                    * ((normalizeTime) ? normalize(timeshift, timeStampDeviation, timeFactor) : timeshift);
        }
        return scaledInput;
    }

    /**
     * an execute once block which first computes the multipliers for normalization
     * and then processes each of the stored inputs
     */
    protected void dischargeInitial(RandomCutForest forest) {
        Deviation tempTimeDeviation = new Deviation();
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            tempTimeDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double timeFactor = tempTimeDeviation.getDeviation();
        double[] factors = null;
        if (requireInitialSegment(false, transformMethod)) {
            Deviation[] tempList = new Deviation[inputLength];
            for (int j = 0; j < inputLength; j++) {
                tempList[j] = new Deviation(deviationList[j].getDiscount());
            }
            for (int i = 0; i < initialValues.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    double value;
                    if (transformMethod == TransformMethod.NORMALIZE) {
                        value = initialValues[i][j];
                    } else {
                        value = (i == 0) ? 0 : initialValues[i][j] - initialValues[i - 1][j];
                    }
                    tempList[j].update(value);
                }
            }
            factors = new double[inputLength];
            for (int j = 0; j < inputLength; j++) {
                factors[j] = tempList[j].getDeviation();
            }
        }

        for (int i = 0; i < valuesSeen; i++) {
            double[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], factors, timeFactor);
            if (internalTimeStamp > 0) {
                timeStampDeviation.update(initialTimeStamps[i] - previousTimeStamps[shingleSize - 1]);
            }
            updateState(initialValues[i], scaledInput, initialTimeStamps[i]);
            forest.update(scaledInput);
        }

        initialTimeStamps = null;
        initialValues = null;
    }

    /**
     * maps a value shifted to the current mean or to a relative space
     *
     * @param value     input value of dimension
     * @param deviation statistic
     * @return the normalized value
     */
    protected double normalize(double value, Deviation deviation, double factor) {
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
     * applies transformations if desired
     *
     * @param inputPoint input point
     * @return a differenced version of the input
     */
    protected double[] transformValues(double[] inputPoint, double[] factors) {
        if (!requireInitialSegment(false, transformMethod)) {
            return super.transformValues(inputPoint);
        }
        double[] input = new double[inputPoint.length];
        if (transformMethod == TransformMethod.NORMALIZE) {
            for (int i = 0; i < input.length; i++) {
                input[i] = weights[i] * normalize(inputPoint[i], deviationList[i], (factors == null) ? 0 : factors[i]);
            }
        } else if (transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
            for (int i = 0; i < input.length; i++) {
                double value = (internalTimeStamp == 0) ? 0
                        : inputPoint[i] - lastShingledInput[lastShingledInput.length - inputLength + i];
                input[i] = weights[i] * normalize(value, deviationList[i], (factors == null) ? 0 : factors[i]);
            }
        }
        return input;
    }
}
