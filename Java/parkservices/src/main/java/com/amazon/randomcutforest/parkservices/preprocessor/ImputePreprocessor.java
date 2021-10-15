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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.LINEAR;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.config.TransformMethod.DIFFERENCE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE_DIFFERENCE;
import static com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest.MINIMUM_OBSERVATIONS_FOR_EXPECTED;

@Getter
@Setter
public class ImputePreprocessor extends InitialSegmentPreprocessor {

    public ImputationMethod DEFAULT_RCF_IMPUTATION_FOR_LOW_DATA = PREVIOUS;
    public ImputationMethod DEFAULT_RCF_IMPUTATION_FOR_INITIAL = LINEAR;

    ThresholdedRandomCutForest thresholdedRandomCutForest;

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
     * @return a scaled/transformed input which can be used in the forest
     */
    protected double[] getScaledInput(double[] input, long timestamp, double[] defaultFactors,
            double defaultTimeFactor) {
        return transformValues(input, defaultFactors);
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
        return preProcess(inputPoint,timestamp,forest,0,null);
    }

    public double[] preProcess(double[] inputPoint, long timestamp, RandomCutForest forest,long lastAnomalyTimeStamp, double[] lastExpectedValue) {

        if (valuesSeen < startNormalization) {
            storeInitial(inputPoint, timestamp);
            return null;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        lastActualInternal  = internalTimeStamp;
        lastAnomalyTimeStamp = previousTimeStamps[shingleSize-1];

        if (mode == ForestMode.STREAMING_IMPUTE) {
            checkArgument(valuesSeen == 0 || timestamp > previousTimeStamps[forest.getShingleSize() - 1],
                    "incorrect order of time");
            // it may make sense to use dataQuality here as well, but that can set runaway
            // effects
            applyTransductiveImpute(imputationMethod,inputPoint, timestamp, timeStampDeviation.getMean(), forest,lastAnomalyTimeStamp,lastExpectedValue);
        }

        return getScaledInput(inputPoint, timestamp, null, 0);
    }

    @Override
    public int verifyActual(int index) {
        if (index == 0) {
            return 0;
        }
        checkArgument(index > -shingleSize, "illegal index");

        int j = shingleSize - 1 + index;
        while ((j < shingleSize - 1) && (previousTimeStamps[j + 1] == previousTimeStamps[j])) {
            ++j;
        }

        return j - shingleSize + 1;
    }

    @Override
    protected void updateTimestamps(long timestamp) {
        if (previousTimeStamps[0] ==  previousTimeStamps[1]) {
            numberOfImputed = numberOfImputed-1;
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
        return (fraction < useImputedFraction && valuesSeen >= shingleSize );
    }

    public AnomalyDescriptor postProcess(AnomalyDescriptor result, double[] inputPoint, long timestamp, RandomCutForest forest, double[] point, double[] scaledInput) {

        if (result.getAnomalyGrade()>0) {
            double[] reference = inputPoint;
            double[] newPoint = result.getExpectedRCFPoint();

            int index = result.getRelativeIndex();

            if (newPoint != null) {
                if (index < 0 && result.isStartOfAnomaly()) {
                    reference = getShingledInput(shingleSize + index);
                    result.setOldValues(reference);
                    result.setOldTimeStamp(getTimeStamp(shingleSize - 1 + index));
                }
                double[] values = getExpectedValue(index, reference, point,
                        newPoint);
                result.setExpectedValues(0, values, 1.0);
            }

            addRelevantAttribution(result);
        }

        if (timeStampDeviation != null) {
            timeStampDeviation.update(timestamp - lastInputTimeStamp);
        }
        ++valuesSeen;
        updateState(inputPoint, scaledInput, timestamp);
        if (updateAllowed()) {
            forest.update(lastShingledPoint);
        }
        return result;
    }


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

        Arrays.fill(previousTimeStamps,initialTimeStamps[0]);
        for (int i = 0; i < valuesSeen; i++) {
            // initial imputation
            lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
            ImputationMethod method = (imputationMethod == RCF)?DEFAULT_RCF_IMPUTATION_FOR_INITIAL:imputationMethod;
            applyTransductiveImpute(method, initialValues[i], initialTimeStamps[i], timeFactor, forest,0,null);
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

    protected boolean useDefault(RandomCutForest forest, int numberToImpute) {
        return (forest.getTotalUpdates() < MINIMUM_OBSERVATIONS_FOR_EXPECTED || dimension < 4 || numberToImpute > 1);
    }

    protected void applyTransductiveImpute(ImputationMethod method, double[] input, long timestamp, double averageGap, RandomCutForest forest, long lastAnomalyTimeStamp, double [] lastExpectedValue) {
        checkArgument(shingleSize > 1, "imputation is not meaningful with shingle size 1");
        int baseDimension = dimension / shingleSize;
        checkArgument(input.length == baseDimension, "error in length");
        if (internalTimeStamp > 1) {
            // the last entry would correspond to a previous actual point
            double gap = (timestamp - previousTimeStamps[shingleSize - 1]) / averageGap;

            if (gap >= 1.5) {
                // note internalTimeStamp changes after postProcess
                if (lastAnomalyTimeStamp == lastActualInternal - 1 && lastExpectedValue != null) {
                    // we choose to not use the last anomaly in further interpolation
                    // we switch the input sequence
                    checkArgument(lastExpectedValue.length == dimension, "incorrect expected point length");
                    lastShingledPoint=Arrays.copyOf(lastExpectedValue,dimension);
                    for(int i=0;i<baseDimension;i++) {
                        lastShingledInput[dimension - baseDimension + i] = inverseTransform(lastExpectedValue[dimension-baseDimension+i], i, -1);
                    }
                }
                /**
                 * note that for a drop rate r, the gap woukd evaluate to 2(1-r) for a simgle missing observation
                 * the 1.4 hardcodes the fact that at most 0.25 fraction of the data can be dropped
                 * the numberOfImputes can be refined to a better estimate
                 *
                 */
                int numberToImpute = (int) Math.ceil(gap) - 1;
                double step = 1.0 /(numberToImpute + 1);

                // the last impute corresponds to the current observed value
                for (int i = 0; i < numberToImpute; i++) {
                    double[] previous = new double[baseDimension];
                    double [] result = null;
                    System.arraycopy(lastShingledInput, (shingleSize - 1) * inputLength, previous, 0, baseDimension);
                    if (method == RCF) {
                        if (useDefault(forest,numberToImpute)) {
                            result = impute(step * (i + 1), previous, input, DEFAULT_RCF_IMPUTATION_FOR_LOW_DATA);
                        } else {
                            result = imputeRCF(forest);
                        }
                    } else {
                        result = impute(step * (i + 1), previous, input, method);
                    }
                    double[] scaledInput = getScaledInput(result, 0L);
                    updateShingle(result,scaledInput);
                    updateTimestamps(timestamp);
                    ++internalTimeStamp;
                    numberOfImputed = numberOfImputed + 1;
                    if (updateAllowed()) {
                        forest.update(lastShingledPoint);
                    }
                }
            }
        }
    }

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

    protected double[] imputeRCF(RandomCutForest forest){
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

    protected double[] imputeRCF(int index, int total, double[] input, RandomCutForest forest){
        int baseDimension = inputLength;
        int dimension = forest.getDimensions();
        int[] positions = new int[baseDimension];
        double[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, 2*baseDimension);
        for (int y = 0; y < baseDimension; y++) {
            positions[y] = dimension - 2*baseDimension + y;
        }
        double [] scaledInput = getScaledInput(input,0L);
        if (transformMethod == DIFFERENCE || transformMethod == NORMALIZE_DIFFERENCE){
            for(int i = 0;i <baseDimension;i++){
                scaledInput[i] = index * scaledInput[i]/total;
            }
        }
        copyAtEnd(temp,scaledInput);
        double[] newPoint = forest.imputeMissingValues(temp, baseDimension, positions);
        return getExpectedValue(baseDimension, dimension - 2*baseDimension, newPoint);
    }
}
