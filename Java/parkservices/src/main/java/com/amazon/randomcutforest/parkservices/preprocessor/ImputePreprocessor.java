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
import static com.amazon.randomcutforest.config.ImputationMethod.NEXT;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;

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
     * a function to determine the number of missing values
     * 
     * @param timestamp the timestamp of the next point
     * @return the number of missing values
     */
    protected int determineNumberOfImputes(long timestamp) {
        return determineGap(timestamp, timeStampDeviation.getMean()) - 1;
    }

    /**
     * an extension of the basec preprocessing
     * 
     * @param description description of the computation for the cirrent point
     * @param forest      the resident RCF
     * @return adds the RCFPoint as well as number of imputed points; it also saves
     *         the last anomaly point and the last anomaly timestamp for later use
     */
    @Override
    public AnomalyDescriptor preProcess(AnomalyDescriptor description, RandomCutForest forest) {
        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        double[] inputPoint = description.getCurrentInput();
        long timestamp = description.getInputTimestamp();

        if (valuesSeen < startNormalization) {
            storeInitial(inputPoint, timestamp);
            return description;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        tempLastAnomalyTimeStamp = description.getLastAnomalyInternalTimestamp();
        double[] lastExpectedPoint = description.getLastExpectedPoint();
        tempLastExpectedValue = (lastExpectedPoint == null) ? null
                : Arrays.copyOf(lastExpectedPoint, lastExpectedPoint.length);
        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        checkArgument(timestamp > lastInputTimeStamp, "incorrect ordering of time");

        // genertate next tuple without changing the forest
        double[] point = generateShingle(inputPoint, timestamp, timeStampDeviation.getMean(), null, false, forest);

        if (point == null) {
            return description;
        }

        description.setRCFPoint(point);
        description.setNumberOfImputes(determineNumberOfImputes(timestamp));
        description.setInternalTimeStamp(internalTimeStamp + description.getNumberOfImputes());
        return description;
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
        return (fraction < useImputedFraction && internalTimeStamp >= shingleSize);
    }

    /**
     * The postprocessing now has to handle imputation while changing the state
     * 
     * @param result the descriptor of the evaluation on the current point
     * @param forest the resident RCF
     * @return the description with the explanation added and state updated
     */
    @Override
    public AnomalyDescriptor postProcess(AnomalyDescriptor result, RandomCutForest forest) {

        double[] point = result.getRCFPoint();
        if (point == null) {
            return result;
        }

        if (result.getAnomalyGrade() > 0 && numberOfImputed == 0) {
            // we cannot predict expected value easily if there are gaps in the shingle
            addRelevantAttribution(result);
        }

        double[] inputPoint = result.getCurrentInput();
        long timestamp = result.getInputTimestamp();

        // generate shingles and commit the results
        generateShingle(inputPoint, timestamp, timeStampDeviation.getMean(), null, true, forest);
        ++valuesSeen;
        return result;
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
            // generate shingles and commit them
            generateShingle(initialValues[i], initialTimeStamps[i], timeFactor, factors, true, forest);
        }

        initialTimeStamps = null;
        initialValues = null;
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
        if (internalTimeStamp <= 1) {
            return 1;
        } else {
            double gap = (timestamp - lastInputTimeStamp) / averageGap;
            return (gap >= 1.5) ? (int) Math.ceil(gap) : 1;
        }
    }

    /**
     * a single function that constructs the next shingle, with the option of
     * committing them to the forest However the shingle needs to be genrated before
     * we process a point; and can only be committed once the point has been scored.
     * Having the same deterministic transformation is essential
     * 
     * @param input        the input point
     * @param timestamp    the input timestamp
     * @param averageGap   the gap in timestamps
     * @param factors      the factors in normalization (not in use after initial
     *                     segment)
     * @param changeForest boolean determining if we commit to the forest or not
     * @param forest       the resident RCF
     * @return the next shingle
     */
    protected double[] generateShingle(double[] input, long timestamp, double averageGap, double[] factors,
            boolean changeForest, RandomCutForest forest) {
        double[] tempShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        if (internalTimeStamp > 0) {
            double[] previous = new double[inputLength];
            System.arraycopy(lastShingledInput, lastShingledInput.length - inputLength, previous, 0, inputLength);
            int numberToImpute = determineGap(timestamp, averageGap) - 1;
            if (numberToImpute > 0) {
                double step = 1.0 / (numberToImpute + 1);
                // the last impute corresponds to the current observed value
                for (int i = 0; i < numberToImpute; i++) {
                    double[] result = impute(step * (i + 1), previous, input, imputationMethod);
                    double[] scaledInput = transformValues(result, factors);
                    if (changeForest) {
                        updateShingle(result, scaledInput);
                        updateTimestamps(timestamp);
                        numberOfImputed = numberOfImputed + 1;
                        if (updateAllowed()) {
                            forest.update(lastShingledPoint);
                        }
                    } else {
                        shiftLeft(tempShingle, inputLength);
                        copyAtEnd(tempShingle, transformValues(result, null));
                    }
                }
            }
        }
        double[] scaledInput = transformValues(input, factors);
        if (changeForest) {
            timeStampDeviation.update(timestamp - lastInputTimeStamp);
            updateState(input, scaledInput, timestamp);
            if (updateAllowed()) {
                forest.update(lastShingledPoint);
            }
            // current shingle
            tempShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        } else {
            shiftLeft(tempShingle, inputLength);
            copyAtEnd(tempShingle, transformValues(input, null));
        }
        return tempShingle;
    }

    /**
     * a simple function that performs a single step imputation in the input space
     * the function has to be deterministic since it is run twice, first at scoring
     * and then at committing to the RCF
     * 
     * @param stepFraction the interpolant fraction
     * @param previous     the previous input point
     * @param input        the current input point
     * @param method       the inputation method of choice
     * @return the imputed/interpolated result
     */
    protected double[] impute(double stepFraction, double[] previous, double[] input, ImputationMethod method) {
        int baseDimension = input.length;
        double[] result = new double[baseDimension];

        if (method == FIXED_VALUES) {
            System.arraycopy(defaultFill, 0, result, 0, baseDimension);
        } else if (method == LINEAR || method == RCF) {
            for (int z = 0; z < input.length; z++) {
                result[z] = previous[z] + stepFraction * (input[z] - previous[z]);
            }
        } else if (method == PREVIOUS) {
            System.arraycopy(previous, 0, result, 0, baseDimension);
        } else if (method == NEXT) {
            System.arraycopy(input, 0, result, 0, baseDimension);
        }
        return result;
    }
}
