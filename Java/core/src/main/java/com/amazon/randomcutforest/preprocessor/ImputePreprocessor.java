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

package com.amazon.randomcutforest.preprocessor;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
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
import com.amazon.randomcutforest.statistics.Deviation;

@Getter
@Setter
public class ImputePreprocessor extends InitialSegmentPreprocessor {

    public static ImputationMethod DEFAULT_INITIAL = LINEAR;
    public static ImputationMethod DEFAULT_DYNAMIC = PREVIOUS;

    /**
     * the builder initializes the numberOfImputed, which is not used in the other
     * classes
     * 
     * @param builder a builder for Preprocessor
     */
    public ImputePreprocessor(Builder<?> builder) {
        super(builder);
        numberOfImputed = shingleSize;
    }

    public float[] getScaledShingledInput(double[] inputPoint, long timestamp, int[] missing, RandomCutForest forest) {
        if (valuesSeen < startNormalization) {
            return null;
        }
        checkArgument(timestamp > previousTimeStamps[shingleSize - 1], "incorrect ordering of time");

        // generate next tuple without changing the forest, these get modified in the
        // transform
        // a primary culprit is differencing, a secondary culprit is the numberOfImputed
        long[] savedTimestamps = Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
        double[] savedShingledInput = Arrays.copyOf(lastShingledInput, lastShingledInput.length);
        float[] savedShingle = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        int savedNumberOfImputed = numberOfImputed;
        int lastActualInternal = internalTimeStamp;

        float[] point = generateShingle(inputPoint, timestamp, missing, getTimeFactor(timeStampDeviations[1]), false,
                forest);

        // restore state
        internalTimeStamp = lastActualInternal;
        numberOfImputed = savedNumberOfImputed;
        previousTimeStamps = Arrays.copyOf(savedTimestamps, savedTimestamps.length);
        lastShingledInput = Arrays.copyOf(savedShingledInput, savedShingledInput.length);
        lastShingledPoint = Arrays.copyOf(savedShingle, savedShingle.length);

        return point;
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
        if (fraction > 1) {
            fraction = 1;
        }
        if (numberOfImputed >= shingleSize - 1 && previousTimeStamps[0] != previousTimeStamps[1]
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

        dataQuality[0].update(1 - fraction);
        return (fraction < useImputedFraction && internalTimeStamp >= shingleSize);
    }

    @Override
    protected void updateTimestamps(long timestamp) {
        /*
         * For imputations done on timestamps other than the current one (specified by
         * the timestamp parameter), the timestamp of the imputed tuple matches that of
         * the input tuple, and we increment numberOfImputed. For imputations done at
         * the current timestamp (if all input values are missing), the timestamp of the
         * imputed tuple is the current timestamp, and we increment numberOfImputed.
         *
         * To check if imputed values are still present in the shingle, we use the
         * condition (previousTimeStamps[0] == previousTimeStamps[1]). This works
         * because previousTimeStamps has a size equal to the shingle size and is filled
         * with the current timestamp.
         *
         * For example, if the last 10 values were imputed and the shingle size is 8,
         * the condition will most likely return false until all 10 imputed values are
         * removed from the shingle.
         *
         * However, there are scenarios where we might miss decrementing
         * numberOfImputed:
         *
         * 1. Not all values in the shingle are imputed. 2. We accumulated
         * numberOfImputed when the current timestamp had missing values.
         *
         * As a result, this could cause the data quality measure to decrease
         * continuously since we are always counting missing values that should
         * eventually be reset to zero. To address the issue, we add code in method
         * updateForest to decrement numberOfImputed when we move to a new timestamp,
         * provided there is no imputation. This ensures the imputation fraction does
         * not increase as long as the imputation is continuing. This also ensures that
         * the forest update decision, which relies on the imputation fraction,
         * functions correctly. The forest is updated only when the imputation fraction
         * is below the threshold of 0.5.
         *
         * Also, why can't we combine the decrement code between updateTimestamps and
         * updateForest together? This would cause Consistency.ImputeTest to fail when
         * testing with and without imputation, as the RCF scores would not change. The
         * method updateTimestamps is used in other places (e.g., updateState and
         * dischargeInitial), not only in updateForest.
         */
        if (previousTimeStamps[0] == previousTimeStamps[1]) {
            numberOfImputed = numberOfImputed - 1;
        }
        super.updateTimestamps(timestamp);
    }

    /**
     * the following function mutates the forest, the lastShingledPoint,
     * lastShingledInput as well as previousTimeStamps, and adds the shingled input
     * to the forest (provided it is allowed by the number of imputes and the
     * transformation function)
     * 
     * @param input          the input point (can be imputed)
     * @param timestamp      the input timestamp (will be the most recent timestamp
     *                       for imputes)
     * @param forest         the resident RCF
     * @param isFullyImputed is the current input fully imputed based on timestamps
     */
    void updateForest(boolean changeForest, double[] input, long timestamp, RandomCutForest forest,
            boolean isFullyImputed) {
        float[] scaledInput = transformer.transformValues(internalTimeStamp, input, getShingledInput(shingleSize - 1),
                null, clipFactor);

        updateShingle(input, scaledInput);
        updateTimestamps(timestamp);
        if (isFullyImputed) {
            // The numImputed is now capped at the shingle size to ensure that the impute
            // fraction,
            // calculated as numberOfImputed * 1.0 / shingleSize, does not exceed 1.
            numberOfImputed = Math.min(numberOfImputed + 1, shingleSize);
        } else if (numberOfImputed > 0) {
            // Decrement numberOfImputed when the new value is not imputed
            numberOfImputed = numberOfImputed - 1;
        }
        if (changeForest) {
            if (forest.isInternalShinglingEnabled()) {
                // update allowed = not updateShingleOnly
                forest.update(scaledInput, !updateAllowed());
            } else if (updateAllowed()) {
                forest.update(lastShingledPoint);
            }
        }
    }

    @Override
    public void update(double[] point, float[] rcfPoint, long timestamp, int[] missing, RandomCutForest forest) {
        if (valuesSeen < startNormalization) {
            storeInitial(point, timestamp, missing); // will change valuesSeen
            if (valuesSeen == startNormalization) {
                dischargeInitial(forest);
            }
            return;
        }
        generateShingle(point, timestamp, missing, getTimeFactor(timeStampDeviations[1]), true, forest);
        // The confidence formula depends on numImputed (the number of recent
        // imputations seen)
        // and seenValues (all values seen). To ensure confidence decreases when
        // numImputed increases,
        // we need to count only non-imputed values as seenValues.
        if (missing == null || missing.length != point.length) {
            ++valuesSeen;
        }
    }

    protected double getTimeFactor(Deviation deviation) {
        double timeFactor = deviation.getMean();
        double dev = deviation.getDeviation();
        if (dev > 0 && dev < timeFactor / 2) {
            // a correction
            timeFactor -= dev * dev / (2 * timeFactor);
        }
        return timeFactor;
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
        double timeFactor = getTimeFactor(tempTimeDeviation);

        prepareInitialInput();
        Deviation[] deviations = getInitialDeviations();
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
                        float[] scaledInput = transformer.transformValues(internalTimeStamp, result,
                                getShingledInput(shingleSize - 1), deviations, clipFactor);
                        updateShingle(result, scaledInput);
                        updateTimestamps(initialTimeStamps[i]);
                        numberOfImputed = numberOfImputed + 1;
                        if (forest.isInternalShinglingEnabled()) {
                            // updateAllowed = not updateShingleOnly
                            forest.update(scaledInput, !updateAllowed());
                        } else {
                            if (updateAllowed()) {
                                forest.update(lastShingledPoint);
                            }
                        }
                    }
                }
            }
            float[] scaledInput = transformer.transformValues(internalTimeStamp, initialValues[i],
                    getShingledInput(shingleSize - 1), deviations, clipFactor);
            // note that initial values are all interpolated by 0,fixed, or linear
            // there are no missing values to handle
            updateState(initialValues[i], scaledInput, initialTimeStamps[i], lastInputTimeStamp, null);
            if (forest.isInternalShinglingEnabled()) {
                // updateAllowed = not updateShingleOnly
                forest.update(scaledInput, !updateAllowed());
            } else {
                if (updateAllowed()) {
                    forest.update(lastShingledPoint);
                }
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

    public int numberOfImputes(long timestamp) {
        long lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        return determineGap(timestamp - lastInputTimeStamp, getTimeFactor(timeStampDeviations[1])) - 1;
    }

    /**
     * a single function that constructs the next shingle, with the option of
     * committing them to the forest However the shingle needs to be generated
     * before we process a point; and can only be committed once the point has been
     * scored. Having the same deterministic transformation can be useful. Note for
     * this imputation timestamp cannot be missing
     *
     * @param averageGap   the gap in timestamps
     * @param changeForest boolean determining if we commit to the forest or not
     * @param forest       the resident RCF
     * @return the next shingle
     */
    protected float[] generateShingle(double[] inputTuple, long timestamp, int[] missingValues, double averageGap,
            boolean changeForest, RandomCutForest forest) {
        long lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        double[] input = Arrays.copyOf(inputTuple, inputLength);
        double[] previous = getShingledInput(shingleSize - 1);
        double[] savedInput = Arrays.copyOf(previous, inputLength);
        int numberToImpute = determineGap(timestamp - lastInputTimeStamp, averageGap) - 1;

        if (imputationMethod != RCF || !forest.isOutputReady()) {
            ImputationMethod method = (imputationMethod == RCF) ? DEFAULT_DYNAMIC : imputationMethod;
            // for STREAMING_IMPUTE the timestamp cannot be missing
            // hence missingValues[] can be 0 to inputLength - 1
            // for next and Linear there are no current values
            // we are forced to use fixedvalues or previous
            if (missingValues != null) {
                for (int missingValue : missingValues) {
                    input[missingValue] = (defaultFill == null) ? previous[missingValue] : defaultFill[missingValue];
                }
            }

            if (numberToImpute > 0) {
                double step = 1.0 / (numberToImpute + 1);
                // the last impute corresponds to the current observed value
                for (int i = 0; i < numberToImpute; i++) {
                    // only the last tuple is partial
                    double[] result = basicImpute(step * (i + 1), previous, input, method);
                    updateForest(changeForest, result, timestamp, forest, true);
                }
            }
        } else {
            // the following is a mechanism to prevent a large number of updates using RCF
            // supposing the data is aggregated at 10min interval and the gap in values
            // correspond to a month = 30 * 24 * 6 imputations -- that would be not only
            // be slow, but also it would be unclear if analysis at shingleSize = 10 is
            // appropriate
            // for imputing 4000+ values. RCF is an example of reinforcement/continuous
            // learning
            // this would be very ripe for hallucination
            // in general, the intent of impute is to correct occasional drops of data
            if (numberToImpute < 3 * shingleSize || !fastForward) {
                for (int i = 0; i < numberToImpute; i++) {
                    double[] result = imputeRCF(forest, null, null);
                    updateForest(changeForest, result, timestamp, forest, true);
                }
            } else {
                // we will skip a lot of values
                double[] shift = getShift(); // uses the transformation to get typical values
                // resets number of imputed
                numberOfImputed = 0;
                for (int i = 0; i < shingleSize - 1; i++) {
                    updateForest(changeForest, shift, timestamp, forest, false);
                }
            }
            // finally the current input may be partial
            if (missingValues != null && missingValues.length > 0) {
                input = imputeRCF(forest, input, missingValues);
            }
        }

        // last parameter isFullyImputed = if we miss everything in inputTuple?
        // This would ensure dataQuality is decreasing if we impute whenever
        updateForest(changeForest, input, timestamp, forest,
                missingValues != null ? missingValues.length == inputTuple.length : false);
        if (changeForest) {
            updateTimeStampDeviations(timestamp, lastInputTimeStamp);
            transformer.updateDeviation(input, savedInput, missingValues);
        }
        return Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
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
        float[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, inputLength);
        int startPosition = inputLength * (shingleSize - 1);
        int[] missingIndices;
        if (partialInput == null) {
            missingIndices = new int[inputLength];
            for (int i = 0; i < inputLength; i++) {
                missingIndices[i] = startPosition + i;
            }
        } else {
            missingIndices = new int[missingValues.length];
            for (int i = 0; i < missingValues.length; i++) {
                missingIndices[i] = startPosition + missingValues[i];
            }
            float[] scaledInput = transformer.transformValues(internalTimeStamp, partialInput,
                    getShingledInput(shingleSize - 1), null, clipFactor);
            copyAtEnd(temp, scaledInput);
        }
        float[] newPoint = forest.imputeMissingValues(temp, missingIndices.length, missingIndices);
        return toDoubleArray(getExpectedBlock(newPoint, 0));
    }
}
