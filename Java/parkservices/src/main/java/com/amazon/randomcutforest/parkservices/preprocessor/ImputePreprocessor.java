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
            storeInitial(description.getCurrentInput(), description.getInputTimestamp());
            return description;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        tempLastAnomalyTimeStamp = description.getLastAnomalyInternalTimestamp();
        double[] lastExpectedPoint = description.getLastExpectedRCFPoint();
        tempLastExpectedValue = (lastExpectedPoint == null) ? null
                : Arrays.copyOf(lastExpectedPoint, lastExpectedPoint.length);
        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        checkArgument(description.getInputTimestamp() > lastInputTimeStamp, "incorrect ordering of time");

        // generate next tuple without changing the forest, these get modified in the
        // transform
        // a primary culprit is differencing, a secondary culprit is the numberOfImputed
        long[] savedTimestamps = Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
        double[] savedShingledInput = Arrays.copyOf(lastShingledInput, lastShingledInput.length);
        double[] savedShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        int savedNumberOfImputed = numberOfImputed;

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
     * fly
     *
     * @return if the forest should be updated
     */
    protected boolean updateAllowed() {
        double fraction = numberOfImputed * 1.0 / (shingleSize);
        if (numberOfImputed == shingleSize - 1 && previousTimeStamps[0] != previousTimeStamps[1]
                && (transformMethod == DIFFERENCE || transformMethod == NORMALIZE_DIFFERENCE)) {
            // this shingle is disconnected from the previously seen values
            // these transformations will have little meaning
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
        double[] scaledInput = transformValues(input, null);
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
     * number of transient shingles (which would noe be admitted to the forest
     * anyways unless there is at least one actual value in the shingle)
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

        double[] factors = getFactors();
        Arrays.fill(previousTimeStamps, initialTimeStamps[0]);
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
                        double[] result = impute(step * (j + 1), previous, initialValues[i], DEFAULT_INITIAL);
                        double[] scaledInput = transformValues(result, factors);
                        updateShingle(result, scaledInput);
                        updateTimestamps(initialTimeStamps[i]);
                        numberOfImputed = numberOfImputed + 1;
                        if (updateAllowed()) {
                            forest.update(lastShingledPoint);
                        }
                    }
                }
            }
            double[] scaledInput = transformValues(initialValues[i], factors);
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
        if (internalTimeStamp > 0) {
            double[] previous = new double[inputLength];
            System.arraycopy(lastShingledInput, lastShingledInput.length - inputLength, previous, 0, inputLength);
            // using the global dependency
            int numberToImpute = determineGap(timestamp - lastInputTimeStamp, averageGap) - 1;
            if (numberToImpute > 0) {
                descriptor.setNumberOfNewImputes(numberToImpute);
                double step = 1.0 / (numberToImpute + 1);
                // the last impute corresponds to the current observed value
                for (int i = 0; i < numberToImpute; i++) {
                    // use a default for RCF if trees are unusable, as reflected in the
                    // isReasonableForecast()
                    ImputationMethod method = descriptor.getImputationMethod();
                    double[] result;
                    if (method == RCF) {
                        if (descriptor.isReasonableForecast()) {
                            result = imputeRCF(forest);
                        } else {
                            result = impute(step * (i + 1), previous, input, DEFAULT_DYNAMIC);
                        }
                    } else {
                        result = impute(step * (i + 1), previous, input, method);
                    }
                    updateForest(changeForest, result, timestamp, forest, true);
                }
            }
        }
        if (changeForest) {
            timeStampDeviation.update(timestamp - lastInputTimeStamp);
            if (deviationList != null) {
                updateDeviation(input);
            }
        }
        updateForest(changeForest, input, timestamp, forest, false);
        return Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
    }

    /**
     * a simple function that performs a single step imputation in the input space
     * the function has to be deterministic since it is run twice, first at scoring
     * and then at committing to the RCF
     * 
     * @param stepFraction the interpolation fraction
     * @param previous     the previous input point
     * @param input        the current input point
     * @param method       the imputation method of choice
     * @return the imputed/interpolated result
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
        } else if (method == NEXT) {
            System.arraycopy(input, 0, result, 0, baseDimension);
        }
        return result;
    }

    /**
     * uses the RCF to impute the next input tuple
     * 
     * @param forest RCF
     * @return the next inout tuple predicted by the RCF
     */
    protected double[] imputeRCF(RandomCutForest forest) {
        double[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, inputLength);
        int startPosition = inputLength * (shingleSize - 1);
        int[] missingIndices = new int[inputLength];
        for (int i = 0; i < inputLength; i++) {
            missingIndices[i] = startPosition + i;
        }
        double[] newPoint = forest.imputeMissingValues(temp, inputLength, missingIndices);
        return invert(inputLength, startPosition, 0, newPoint);
    }
}
