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
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.LINEAR;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest.MINIMUM_OBSERVATIONS_FOR_EXPECTED;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class ImputePreprocessor extends InitialSegmentPreprocessor {

    public ImputationMethod DEFAULT_RCF_IMPUTATION_FOR_LOW_DATA = PREVIOUS;
    public ImputationMethod DEFAULT_RCF_IMPUTATION_FOR_INITIAL = LINEAR;

    ThresholdedRandomCutForest thresholdedRandomCutForest;

    double[] tempLastExpectedValue;
    long tempLastAnomalyTimeStamp;

    /**
     * the builder initializes the numberOfImputed, which is not used in the other
     * classes
     * 
     * @param builder a builder for Preprocessor
     */
    public ImputePreprocessor(Builder<?> builder) {
        super(builder);
        thresholdedRandomCutForest = builder.thresholdedRandomCutForest;
        numberOfImputed = shingleSize;
    }

    /**
     * given an input produces a scaled transform to be used in the forest
     *
     * @param input     the actual input seen
     * @param timestamp timestamp of said input
     * @return a scaled/transformed input which has the same length os input
     */
    protected double[] getScaledInput(double[] input, long timestamp, double[] defaultFactors,
            double defaultTimeFactor) {
        checkArgument(valuesSeen == 0 || timestamp >= previousTimeStamps[shingleSize - 1], "incorrect order of time");
        return transformValues(input, defaultFactors);
    }

    public double[] preProcess(double[] inputPoint, long timestamp, RandomCutForest forest, long lastAnomalyTimeStamp,
            double[] lastExpectedValue) {

        if (valuesSeen < startNormalization) {
            storeInitial(inputPoint, timestamp);
            return null;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        tempLastAnomalyTimeStamp = lastAnomalyTimeStamp;
        tempLastExpectedValue = (lastExpectedValue == null) ? null
                : Arrays.copyOf(lastExpectedValue, lastExpectedValue.length);
        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        checkArgument(timestamp > lastInputTimeStamp, "incorrect ordering of time");

        return getImputedShingle(inputPoint, timestamp, timeStampDeviation.getMean(), forest);
    }

    /**
     * the timestamps are now used to calculated the number of imputed tuples in the
     * shingle
     * 
     * @param timestamp the timestamp of the current input
     */
    @Override
    protected void updateTimestamps(long timestamp) {
        if (previousTimeStamps[0] == previousTimeStamps[1]) {
            numberOfImputed = numberOfImputed - 1;
        }

        for (int i = 0; i < shingleSize - 1; i++) {
            previousTimeStamps[i] = previousTimeStamps[i + 1];
        }
        previousTimeStamps[shingleSize - 1] = timestamp;
        ++internalTimeStamp;
    }

    /**
     * decides if the forest should be updated, this is needed for imputation on the
     * fly
     *
     * @return if the forest should be updated
     */
    protected boolean updateAllowed() {
        double fraction = numberOfImputed * 1.0 / (shingleSize);
        dataQuality.update(1 - fraction);
        return (fraction < useImputedFraction && valuesSeen >= shingleSize);
    }

    /**
     * The postprocessing now has to handle imputation while changing the state
     * 
     * @param result     the descriptor of the evaluation on the current point
     * @param inputPoint the current input point
     * @param timestamp  the timestamp of the current input
     * @param forest     the resident RCF
     * @return
     */
    @Override
    public AnomalyDescriptor postProcess(AnomalyDescriptor result, double[] inputPoint, long timestamp,
            RandomCutForest forest) {

        double[] point = result.getRcfPoint();

        int gap = determineGap(timestamp, timeStampDeviation.getMean()) - 1;
        if (result.getAnomalyGrade() > 0) {
            double[] reference = inputPoint;
            double[] newPoint = result.getExpectedRCFPoint();

            int index = Math.min(result.getRelativeIndex() + gap, 0);
            result.setRelativeIndex(index);

            if (newPoint != null) {
                if (index < 0 && result.isStartOfAnomaly()) {
                    reference = getShingledInput(shingleSize + index);
                    result.setOldValues(reference);
                    result.setOldTimeStamp(getTimeStamp(shingleSize - 1 + index));
                }
                double[] values = getExpectedValue(index, reference, point, newPoint);
                result.setExpectedValues(0, values, 1.0);
            }

            addRelevantAttribution(result);
        }

        if (gap > 0) {
            // if the linear interpolation did not produce an anomaly then accept those
            if (result.getAnomalyGrade() > 0) {
                applyTransductiveImpute(imputationMethod, inputPoint, timestamp, timeStampDeviation.getMean(), forest,
                        tempLastAnomalyTimeStamp, tempLastExpectedValue);
            } else {
                applyTransductiveImpute(LINEAR, inputPoint, timestamp, timeStampDeviation.getMean(), forest,
                        tempLastAnomalyTimeStamp, tempLastExpectedValue);
            }
        }

        updateState(inputPoint, point, timestamp);
        if (timeStampDeviation != null) {
            timeStampDeviation.update(timestamp - lastInputTimeStamp);
        }
        ++valuesSeen;
        if (updateAllowed()) {
            forest.update(lastShingledPoint);
        }
        return result;
    }

    /**
     * a simplified version of getting the expected values in the original space
     * 
     * @param base          the input dimension
     * @param startPosition the starting position in the shingle
     * @param newPoint      the shingle with the expected values
     * @return the values (at startposition) which would have produced newPoint
     */
    protected double[] getExpectedValue(int base, int startPosition, double[] newPoint) {
        double[] values = new double[base];
        for (int i = 0; i < base; i++) {
            values[i] = inverseTransform(newPoint[startPosition + i], i, 0);
        }
        return values;
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
        double timeFactor = tempTimeDeviation.getMean();

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

        Arrays.fill(previousTimeStamps, initialTimeStamps[0]);
        for (int i = 0; i < valuesSeen; i++) {
            // initial imputation
            lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
            lastActualInternal = internalTimeStamp;
            ImputationMethod method = (imputationMethod == RCF) ? DEFAULT_RCF_IMPUTATION_FOR_INITIAL : imputationMethod;
            applyTransductiveImpute(method, initialValues[i], initialTimeStamps[i], timeFactor, forest, 0, null);
            double[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], factors, timeFactor);
            if (internalTimeStamp > 0) {
                timeStampDeviation.update(initialTimeStamps[i] - lastInputTimeStamp);
            }
            updateState(initialValues[i], scaledInput, initialTimeStamps[i]);
            // update forest
            if (updateAllowed()) {
                forest.update(lastShingledPoint);
            }

        }

        initialTimeStamps = null;
        initialValues = null;
    }

    // a function to determine if a default strategy should be used -- instead of
    // RCF
    // note the condition in the end -- those tuples should probably not be in the
    // forest as well
    // there is little value in imputing those
    protected boolean useDefault(RandomCutForest forest, int numberToImpute) {
        return (forest.getTotalUpdates() < MINIMUM_OBSERVATIONS_FOR_EXPECTED || dimension < 4
                || numberToImpute > shingleSize * useImputedFraction);
    }

    /**
     * determines the gap between the last known timestamp and the current timestamp
     * 
     * @param timestamp  current timestamp
     * @param averageGap the average gap (often determined by
     *                   timeStampDeviation.getMean()
     * @return the number of positions till timestamp
     */
    protected int determineGap(long timestamp, double averageGap) {
        if (internalTimeStamp == 1) {
            return 1;
        } else {
            double gap = (timestamp - lastInputTimeStamp) / averageGap;
            return (gap >= 1.5) ? (int) Math.ceil(gap) : 1;
        }
    }

    /**
     * applies a imputation (which is cognizant of the last actual value)
     * 
     * @param method               the imputation method of choice (unless defaults
     *                             apply)
     * @param input                the input point
     * @param timestamp            timestamp of the current input
     * @param averageGap           the average gap between points
     * @param forest               the resident RCF
     * @param lastAnomalyTimeStamp time stamp of the last anomaly
     * @param lastExpectedValue    the expected value, which should be used in the
     *                             extension
     */
    protected void applyTransductiveImpute(ImputationMethod method, double[] input, long timestamp, double averageGap,
            RandomCutForest forest, long lastAnomalyTimeStamp, double[] lastExpectedValue) {
        checkArgument(shingleSize > 1, "imputation is not meaningful with shingle size 1");
        int baseDimension = dimension / shingleSize;
        checkArgument(input.length == baseDimension, "error in length");
        int numberToImpute = determineGap(timestamp, averageGap) - 1;
        if (numberToImpute > 0) {
            if (lastAnomalyTimeStamp == lastActualInternal - 1 && lastExpectedValue != null) {
                // we choose to not use the last anomaly in further interpolation
                // we switch the input sequence
                checkArgument(lastExpectedValue.length == dimension, "incorrect expected point length");
                lastShingledPoint = Arrays.copyOf(lastExpectedValue, dimension);
                for (int i = 0; i < baseDimension; i++) {
                    lastShingledInput[dimension - baseDimension + i] = inverseTransform(
                            lastExpectedValue[dimension - baseDimension + i], i, -1);
                }
            }

            double step = 1.0 / (numberToImpute + 1);
            // the last impute corresponds to the current observed value
            for (int i = 0; i < numberToImpute; i++) {
                double[] previous = new double[baseDimension];
                double[] result = null;
                System.arraycopy(lastShingledInput, (shingleSize - 1) * inputLength, previous, 0, baseDimension);
                if (method == RCF) {
                    if (useDefault(forest, numberToImpute)) {
                        result = impute(step * (i + 1), previous, input, DEFAULT_RCF_IMPUTATION_FOR_LOW_DATA);
                    } else {
                        result = imputeRCF(forest);
                    }
                } else {
                    result = impute(step * (i + 1), previous, input, method);
                }
                // System.out.println("IMPUTE " + internalTimeStamp);
                double[] scaledInput = getScaledInput(result, 0L);
                updateShingle(result, scaledInput);
                updateTimestamps(timestamp);
                numberOfImputed = numberOfImputed + 1;
                if (updateAllowed()) {
                    forest.update(lastShingledPoint);
                }
            }
        }
    }

    /**
     * a linear interpolator which is used to score a point but NOT to add to the
     * forest
     * 
     * @param input      input point
     * @param timestamp  timestamp of the input point
     * @param averageGap the average gap seen so far
     * @param forest     the resident RCF
     * @return a shingle corresponding to the input point -- the forest is not
     *         updated, but \ this shingle is needed for scoring
     */

    protected double[] getImputedShingle(double[] input, long timestamp, double averageGap, RandomCutForest forest) {
        int baseDimension = dimension / shingleSize;
        double[] tempShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        double[] transformedInput = transformValues(input, null);
        int numberToImpute = determineGap(timestamp, averageGap) - 1;
        double step = 1.0 / (numberToImpute + 1);
        double[] previous = new double[baseDimension];
        System.arraycopy(lastShingledPoint, lastShingledPoint.length - inputLength, previous, 0, baseDimension);
        double[] result = new double[baseDimension];
        for (int i = 0; i < numberToImpute + 1; i++) {
            for (int z = 0; z < input.length; z++) {
                result[z] = (1 - step) * previous[z] + step * transformedInput[z];
            }
            shiftLeft(tempShingle, inputLength);
            copyAtEnd(tempShingle, result);
        }
        return tempShingle;
    }

    /**
     * a simple impute function based on the methods
     */

    protected double[] impute(double stepFraction, double[] previous, double[] input, ImputationMethod method) {
        int baseDimension = input.length;
        double[] result = new double[baseDimension];

        if (method == FIXED_VALUES) {
            System.arraycopy(defaultFill, 0, result, 0, baseDimension);
        } else if (method == LINEAR) {
            for (int z = 0; z < input.length; z++) {
                result[z] = previous[z] + stepFraction * (input[z] - previous[z]);
            }
        } else if (method == PREVIOUS) {
            System.arraycopy(previous, 0, result, 0, baseDimension);
        }
        return result;
    }

    /**
     * using extrapolation via the RCF -- note that it is independent of the input
     * point
     * 
     * @param forest the resident RCF
     * @return the next input (can be multidimensional) -- this is likely to be
     *         noisy for low shingle sizes
     */
    protected double[] imputeRCF(RandomCutForest forest) {
        int baseDimension = inputLength;
        int dimension = forest.getDimensions();
        int[] positions = new int[baseDimension];
        double[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, baseDimension);
        for (int y = 0; y < baseDimension; y++) {
            positions[y] = dimension - baseDimension + y;
        }
        double[] newPoint = forest.imputeMissingValues(temp, baseDimension, positions);
        return getExpectedValue(baseDimension, dimension - baseDimension, newPoint);
    }

}
