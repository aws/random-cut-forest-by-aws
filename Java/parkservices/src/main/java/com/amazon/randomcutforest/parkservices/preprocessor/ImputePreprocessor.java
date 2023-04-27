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
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.config.TransformMethod.DIFFERENCE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE_DIFFERENCE;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.IRCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class ImputePreprocessor extends InitialSegmentPreprocessor {

    public static ImputationMethod DEFAULT_INITIAL = LINEAR;
    public static ImputationMethod DEFAULT_DYNAMIC = PREVIOUS;

    ThresholdedRandomCutForest thresholdedRandomCutForest;

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
     * stores initial data for normalization
     *
     * @param inputPoint input data
     * @param timestamp  timestamp
     */
    protected void storeInitial(double[] inputPoint, long timestamp, int[] missingValues) {
        initialTimeStamps[valuesSeen] = timestamp;
        checkArgument(inputPoint.length == inputLength, "incorrect length");
        checkArgument(missingValues == null || missingValues.length <= inputLength, "unusual missing values list");
        int length = inputLength + ((missingValues == null) ? 0 : missingValues.length);
        double[] temp = new double[length];
        System.arraycopy(inputPoint, 0, temp, 0, inputLength);
        if (missingValues != null) {
            for (int i = 0; i < length - inputLength; i++) {
                temp[inputLength + i] = missingValues[i];
            }
        }
        initialValues[valuesSeen] = temp;
        ++valuesSeen;
    }

    /**
     * prepare initial values which can have missing entries in individual tuples.
     * We use a simple interpolation strategy. At some level, lack of data simply
     * cannot be solved easily without data. This is run as one of the initial steps
     * in dischargeInitial() If all the entries corresponding to some variables are
     * missing -- there is no good starting point; we assume the value is 0, unless
     * there is a defaultFill()
     */
    void prepareInitialInput() {
        boolean[][] missing = new boolean[initialValues.length][inputLength];
        for (int i = 0; i < initialValues.length; i++) {
            Arrays.fill(missing[i], false);
            int length = initialValues[i].length - inputLength;
            for (int j = 0; j < length; j++) {
                missing[i][(int) initialValues[i][inputLength + j]] = true;
            }
        }
        boolean[] startingValuesSet = new boolean[inputLength];

        if (imputationMethod == ZERO) {
            for (int i = 0; i < initialValues.length - 1; i++) {
                for (int j = 0; j < inputLength; j++) {
                    initialValues[i][j] = (missing[i][j]) ? initialValues[i][j] : 0;
                }
            }
        } else if (imputationMethod == FIXED_VALUES || defaultFill != null) {
            for (int i = 0; i < initialValues.length - 1; i++) {
                for (int j = 0; j < inputLength; j++) {
                    initialValues[i][j] = (missing[i][j]) ? initialValues[i][j] : defaultFill[j];
                }
            }
        } else { // no simple alternative other than linear interpolation
            for (int j = 0; j < inputLength; j++) {
                int next = 0;
                while (next < initialValues.length && missing[next][j]) {
                    ++next;
                }
                startingValuesSet[j] = (next < initialValues.length);
                if (startingValuesSet[j]) {
                    initialValues[0][j] = initialValues[next][j];
                    missing[0][j] = false;
                    // note if the first value si present then i==0
                    int start = 0;
                    while (start < initialValues.length - 1) {
                        int end = start + 1;
                        while (end < initialValues.length && missing[end][j]) {
                            ++end;
                        }
                        if (end < initialValues.length && end > start + 1) {
                            for (int y = start + 1; y < end; y++) { // linear interpolation
                                double factor = (1.0 * initialTimeStamps[start] - initialTimeStamps[y])
                                        / (initialTimeStamps[start] - initialTimeStamps[end]);
                                initialValues[y][j] = factor * initialValues[start][j]
                                        + (1 - factor) * initialValues[end][j];
                            }
                        }
                        start = end;
                    }
                } else {
                    // set 0; note there is no value in the entire column.
                    for (int y = 0; y < initialValues.length; y++) {
                        initialValues[y][j] = 0;
                    }
                }
            }
        }
        // truncate to input length, since the missing values were stored as well
        for (int i = 0; i < initialValues.length; i++) {
            initialValues[i] = Arrays.copyOf(initialValues[i], inputLength);
        }
    }

    /**
     * preprocessor that can buffer the initial input as well as impute missing
     * values on the fly note that the forest should not be updated before the point
     * has been scored
     *
     * @param description           description of the input
     * @param lastAnomalyDescriptor the descriptor of the last anomaly
     * @param forest                RCF
     * @return an AnomalyDescriptor used in anomaly detection
     */
    @Override
    public AnomalyDescriptor preProcess(AnomalyDescriptor description, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        initialSetup(description, lastAnomalyDescriptor, forest);

        if (valuesSeen < startNormalization) {
            storeInitial(description.getCurrentInput(), description.getInputTimestamp(),
                    description.getMissingValues());
            return description;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        checkArgument(description.getInputTimestamp() > previousTimeStamps[shingleSize - 1],
                "incorrect ordering of time");

        // generate next tuple without changing the forest, these get modified in the
        // transform
        // a primary culprit is differencing, a secondary culprit is the numberOfImputed
        long[] savedTimestamps = Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
        double[] savedShingledInput = Arrays.copyOf(lastShingledInput, lastShingledInput.length);
        double[] savedShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        int savedNumberOfImputed = numberOfImputed;
        int lastActualInternal = internalTimeStamp;

        double[] point = generateShingle(description, timeStampDeviation.getMean(), false, forest);

        // restore state
        internalTimeStamp = lastActualInternal;
        numberOfImputed = savedNumberOfImputed;
        previousTimeStamps = Arrays.copyOf(savedTimestamps, savedTimestamps.length);
        lastShingledInput = Arrays.copyOf(savedShingledInput, savedShingledInput.length);
        lastShingledPoint = Arrays.copyOf(savedShingle, savedShingle.length);

        if (point == null) {
            return description;
        }

        description.setRCFPoint(point);
        description.setInternalTimeStamp(internalTimeStamp + description.getNumberOfNewImputes());
        return description;
    }

    /**
     * the timestamps are now used to calculate the number of imputed tuples in the
     * shingle
     * 
     * @param timestamp the timestamp of the current input
     */
    @Override
    protected void updateTimestamps(long timestamp) {
        if (previousTimeStamps[0] == previousTimeStamps[1]) {
            numberOfImputed = numberOfImputed - 1;
        }
        super.updateTimestamps(timestamp);
    }

    /**
     * decides if the forest should be updated, this is needed for imputation on the
     * fly. The main goal of this function is to avoid runaway sequences where a
     * single input changes the forest too much. But in some cases that behavior can
     * be warranted and then this function should be changed
     *
     * @return if the forest should be updated
     */
    protected boolean updateAllowed() {
        double fraction = numberOfImputed * 1.0 / (shingleSize);
        if (numberOfImputed == shingleSize - 1 && previousTimeStamps[0] != previousTimeStamps[1]
                && (transformMethod == DIFFERENCE || transformMethod == NORMALIZE_DIFFERENCE)) {
            // this shingle is disconnected from the previously seen values
            // these transformations will have little meaning
            // positions 0 and 1 corresponds to the oldest in the shingle -- if we admit
            // that case
            // then we would admit a shingle where impact of the most recent observation is
            // shingleSize - 1
            // and the oldest one is 1. It seemed conservative to not allow that --
            // primarily to stop a
            // "runaway" effect where a single value (and its imputations affect
            // everything).
            // A gap at positions 1 and 2 would correspond to a shingleSize - 2 and 2 (or
            // two different points).
            return false;
        }
        dataQuality.update(1 - fraction);
        return (fraction < useImputedFraction && internalTimeStamp >= shingleSize);
    }

    /**
     * the following function mutates the forest, the lastShingledPoint,
     * lastShingledInput as well as previousTimeStamps, and adds the shingled input
     * to the forest (provided it is allowed by the number of imputes and the
     * transformation function)
     * 
     * @param input     the input point (can be imputed)
     * @param timestamp the input timestamp (will be the most recent timestamp for
     *                  imputes)
     * @param forest    the resident RCF
     * @param isImputed is the current input imputed
     */
    void updateForest(boolean changeForest, double[] input, long timestamp, RandomCutForest forest, boolean isImputed) {
        double[] scaledInput = transformer.transformValues(internalTimeStamp, input, getShingledInput(shingleSize - 1),
                null, clipFactor);
        updateShingle(input, scaledInput);
        updateTimestamps(timestamp);
        if (isImputed) {
            numberOfImputed = numberOfImputed + 1;
        }
        if (changeForest && updateAllowed()) {
            forest.update(lastShingledPoint);
        }
    }

    /**
     * The postprocessing now has to handle imputation while changing the state;
     * note that the imputation is repeated to avoid storing potentially large
     * number of transient shingles (which would not be admitted to the forest
     * unless there is at least one actual value in the shingle)
     * 
     * @param result                the descriptor of the evaluation on the current
     *                              point
     * @param lastAnomalyDescriptor the descriptor of the last known anomaly
     * @param forest                the resident RCF
     * @return the description with the explanation added and state updated
     */
    @Override
    public AnomalyDescriptor postProcess(AnomalyDescriptor result, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        double[] point = result.getRCFPoint();
        if (point == null) {
            return result;
        }

        if (result.getAnomalyGrade() > 0 && (numberOfImputed == 0 || (result.getTransformMethod() != DIFFERENCE)
                && (result.getTransformMethod() != NORMALIZE_DIFFERENCE))) {
            // we cannot predict expected value easily if there are gaps in the shingle
            // this is doubly complicated for differenced transforms (if there are anu
            // imputations in the shingle)
            addRelevantAttribution(result);
        }

        generateShingle(result, timeStampDeviation.getMean(), true, forest);
        ++valuesSeen;
        return result;
    }

    /**
     * a block which is executed once. It first computes the multipliers for
     * normalization and then processes each of the stored inputs
     */
    protected void dischargeInitial(RandomCutForest forest) {
        Deviation tempTimeDeviation = new Deviation();
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            tempTimeDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double timeFactor = tempTimeDeviation.getMean();

        prepareInitialInput();
        Deviation[] deviations = getDeviations();
        Arrays.fill(previousTimeStamps, initialTimeStamps[0]);
        numberOfImputed = shingleSize;
        for (int i = 0; i < valuesSeen; i++) {
            // initial imputation; not using the global dependency
            long lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
            if (internalTimeStamp > 0) {
                double[] previous = new double[inputLength];
                System.arraycopy(lastShingledInput, lastShingledInput.length - inputLength, previous, 0, inputLength);
                int numberToImpute = determineGap(initialTimeStamps[i] - lastInputTimeStamp, timeFactor) - 1;
                if (numberToImpute > 0) {
                    double step = 1.0 / (numberToImpute + 1);
                    // the last impute corresponds to the current observed value
                    for (int j = 0; j < numberToImpute; j++) {
                        double[] result = basicImpute(step * (j + 1), previous, initialValues[i], DEFAULT_INITIAL);
                        double[] scaledInput = transformer.transformValues(internalTimeStamp, result,
                                getShingledInput(shingleSize - 1), deviations, clipFactor);
                        updateShingle(result, scaledInput);
                        updateTimestamps(initialTimeStamps[i]);
                        numberOfImputed = numberOfImputed + 1;
                        if (updateAllowed()) {
                            forest.update(lastShingledPoint);
                        }
                    }
                }
            }
            double[] scaledInput = transformer.transformValues(internalTimeStamp, initialValues[i],
                    getShingledInput(shingleSize - 1), deviations, clipFactor);
            updateState(initialValues[i], scaledInput, initialTimeStamps[i], lastInputTimeStamp);
            if (updateAllowed()) {
                forest.update(lastShingledPoint);
            }
        }
        initialTimeStamps = null;
        initialValues = null;
    }

    /**
     * determines the gap between the last known timestamp and the current timestamp
     * 
     * @param timestampGap current gap
     * @param averageGap   the average gap (often determined by
     *                     timeStampDeviation.getMean()
     * @return the number of positions till timestamp
     */
    protected int determineGap(long timestampGap, double averageGap) {
        if (internalTimeStamp <= 1) {
            return 1;
        } else {
            double gap = timestampGap / averageGap;
            return (gap >= 1.5) ? (int) Math.ceil(gap) : 1;
        }
    }

    /**
     * a single function that constructs the next shingle, with the option of
     * committing them to the forest However the shingle needs to be generated
     * before we process a point; and can only be committed once the point has been
     * scored. Having the same deterministic transformation is essential
     *
     * @param descriptor   description of the current point
     * @param averageGap   the gap in timestamps
     * @param changeForest boolean determining if we commit to the forest or not
     * @param forest       the resident RCF
     * @return the next shingle
     */
    protected double[] generateShingle(AnomalyDescriptor descriptor, double averageGap, boolean changeForest,
            RandomCutForest forest) {
        double[] input = descriptor.getCurrentInput();
        long timestamp = descriptor.getInputTimestamp();
        long lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        int[] missingValues = descriptor.getMissingValues();
        checkArgument(missingValues == null || (imputationMethod != LINEAR && imputationMethod != NEXT),
                " cannot perform imputation on most recent missing value with this method");
        /*
         * Note only ZERO, FIXED_VALUES, PREVIOUS and RCF are reasonable options if
         * missing values are present.
         */

        checkArgument(internalTimeStamp > 0, "imputation should have forced normalization");
        double[] savedInput = getShingledInput(shingleSize - 1);

        // previous value should be defined
        double[] previous = new double[inputLength];
        System.arraycopy(lastShingledInput, lastShingledInput.length - inputLength, previous, 0, inputLength);
        // using the global dependency
        int numberToImpute = determineGap(timestamp - lastInputTimeStamp, averageGap) - 1;
        if (numberToImpute > 0) {
            descriptor.setNumberOfNewImputes(numberToImpute);
            double step = 1.0 / (numberToImpute + 1);
            // the last impute corresponds to the current observed value
            for (int i = 0; i < numberToImpute; i++) {
                double[] result = impute(false, descriptor, step * (i + 1), previous, forest);
                updateForest(changeForest, result, timestamp, forest, true);
            }
        }
        double[] newInput = (missingValues == null) ? input : impute(true, descriptor, 0, previous, forest);

        updateForest(changeForest, newInput, timestamp, forest, false);
        if (changeForest) {
            timeStampDeviation.update(timestamp - lastInputTimeStamp);
            transformer.updateDeviation(newInput, savedInput);
        }
        return Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
    }

    /**
     * The impute step which predicts the completion of a partial input or predicts
     * the entire input for that timestamp
     * 
     * @param isPartial    a boolean indicating if the input is partial
     * @param descriptor   the state of the current evaluation (missing values
     *                     cannot be null for partial tuples)
     * @param stepFraction the (time) position of the point to impute (can also be
     *                     the final possibly incomplete point)
     * @param previous     the last input
     * @param forest       the RCF
     * @return the imputed tuple for the time position
     */
    protected double[] impute(boolean isPartial, AnomalyDescriptor descriptor, double stepFraction, double[] previous,
            RandomCutForest forest) {
        double[] input = descriptor.getCurrentInput();
        int[] missingValues = descriptor.getMissingValues();
        // we will pass partial input, which would be true for only one tuple
        double[] partialInput = (isPartial) ? Arrays.copyOf(input, inputLength) : null;
        // use a default for RCF if trees are unusable, as reflected in the
        // isReasonableForecast()
        ImputationMethod method = descriptor.getImputationMethod();
        if (method == RCF) {
            if (descriptor.isReasonableForecast()) {
                return imputeRCF(forest, partialInput, missingValues);
            } else {
                return basicImpute(stepFraction, previous, partialInput, DEFAULT_DYNAMIC);
            }
        } else {
            return basicImpute(stepFraction, previous, input, method);
        }
    }

    /**
     * a basic function that performs a single step imputation in the input space
     * the function has to be deterministic since it is run twice, first at scoring
     * and then at committing to the RCF
     * 
     * @param stepFraction the interpolation fraction
     * @param previous     the previous input point
     * @param input        the current input point
     * @param method       the imputation method of choice
     * @return the imputed/interpolated result
     */
    protected double[] basicImpute(double stepFraction, double[] previous, double[] input, ImputationMethod method) {
        double[] result = new double[inputLength];
        if (method == FIXED_VALUES) {
            System.arraycopy(defaultFill, 0, result, 0, inputLength);
        } else if (method == LINEAR) {
            for (int z = 0; z < inputLength; z++) {
                result[z] = previous[z] + stepFraction * (input[z] - previous[z]);
            }
        } else if (method == PREVIOUS) {
            System.arraycopy(previous, 0, result, 0, inputLength);
        } else if (method == NEXT) {
            System.arraycopy(input, 0, result, 0, inputLength);
        }
        return result;
    }

    /**
     * Uses RCF to impute the missing values in the current input or impute the
     * entire set of values for that time step (based on partial input being null)
     * 
     * @param forest        the RCF
     * @param partialInput  the information available about the most recent point
     * @param missingValues the array indicating missing values for the partial
     *                      input
     * @return the potential completion of the partial tuple or the predicted
     *         current value
     */
    protected double[] imputeRCF(RandomCutForest forest, double[] partialInput, int[] missingValues) {
        double[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, inputLength);
        int startPosition = inputLength * (shingleSize - 1);
        int[] missingIndices;
        if (missingValues == null) {
            missingIndices = new int[inputLength];
            for (int i = 0; i < inputLength; i++) {
                missingIndices[i] = startPosition + i;
            }
        } else {
            checkArgument(partialInput != null, "incorrect input");
            missingIndices = Arrays.copyOf(missingValues, missingValues.length);
            double[] scaledInput = transformer.transformValues(internalTimeStamp, partialInput,
                    getShingledInput(shingleSize - 1), null, clipFactor);
            copyAtEnd(temp, scaledInput);
        }
        double[] newPoint = forest.imputeMissingValues(temp, missingIndices.length, missingIndices);
        return invert(inputLength, startPosition, 0, newPoint);
    }
}
