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

import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.preprocessor.transform.WeightedTransformer.NUMBER_OF_STATS;
import static java.lang.Math.round;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.statistics.Deviation;

@Getter
@Setter
public class InitialSegmentPreprocessor extends Preprocessor {

    public InitialSegmentPreprocessor(Builder<?> builder) {
        super(builder);
        initialValues = new double[startNormalization][];
        initialTimeStamps = new long[startNormalization];
    }

    /**
     * stores initial data for normalization. It is possible to perform the
     * imputation inline while storing (for some options) but it seems cleaner to
     * perform en masse imputation (and more complicated algorithms can be used)
     *
     * @param inputPoint    input data
     * @param timestamp     timestamp
     * @param missingValues missing values
     */
    protected void storeInitial(double[] inputPoint, long timestamp, int[] missingValues) {
        // note that timestamps cannot be missing for updates
        initialTimeStamps[valuesSeen] = timestamp;
        int length = inputLength + ((missingValues == null) ? 0 : missingValues.length);
        double[] temp = new double[length];
        System.arraycopy(inputPoint, 0, temp, 0, inputLength);
        if (missingValues != null) {
            for (int i = 0; i < length - inputLength; i++) {
                temp[inputLength + i] = missingValues[i];
            }
        }
        initialValues[valuesSeen] = temp;
        valuesSeen++;
    }

    /**
     * prepare initial values which can have missing entries in individual tuples.
     * We use a simple interpolation strategy. At some level, lack of data simply
     * cannot be solved easily without data. This is run as one of the initial steps
     * in dischargeInitial() If all the entries corresponding to some variables are
     * missing -- there is no good starting point; we assume the value is
     * defaultFill()
     */
    double prepareInitialInput() {
        int totalMissing = 0;
        // note that timestamp cannot be missing for updates
        boolean[][] missing = new boolean[initialValues.length][inputLength];
        for (int i = 0; i < initialValues.length; i++) {
            Arrays.fill(missing[i], false);
            int length = initialValues[i].length - inputLength;
            for (int j = 0; j < length; j++) {
                // duplicates are fine; but should not be encouraged
                ++totalMissing;
                missing[i][(int) round(initialValues[i][inputLength + j])] = true;
            }
        }

        if (imputationMethod == ZERO || imputationMethod == FIXED_VALUES) {
            for (int i = 0; i < initialValues.length - 1; i++) {
                for (int j = 0; j < inputLength; j++) {
                    initialValues[i][j] = (!missing[i][j]) ? initialValues[i][j] : defaultFill[j];
                }
            }
        } else { // no simple alternative other than linear interpolation
                 // at least for the initial segment -- because the trees are
                 // not ready
            boolean[] startingValuesSet = new boolean[inputLength];
            for (int j = 0; j < inputLength; j++) {
                // what is the first is missing?
                int next = 0;
                startingValuesSet[j] = false;
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
                    if (defaultFill != null) {
                        // can be set for other options as well
                        for (int y = 0; y < initialValues.length; y++) {
                            initialValues[y][j] = defaultFill[j];
                        }
                    } else {
                        for (int y = 0; y < initialValues.length; y++) {
                            initialValues[y][j] = 0;
                        }
                    }
                }
            }
        }

        // truncate to input length, since the missing values were stored as well
        for (int i = 0; i < initialValues.length; i++) {
            initialValues[i] = Arrays.copyOf(initialValues[i], inputLength);
        }
        return 1.0 - (1.0 * totalMissing) / initialValues.length;
    }

    @Override
    public void update(double[] point, float[] rcfPoint, long timestamp, int[] missing, RandomCutForest forest) {
        if (valuesSeen < startNormalization) {
            storeInitial(point, timestamp, missing);
            // will change valuesSeen
            if (valuesSeen == startNormalization) {
                dischargeInitial(forest);
            }
            return;
        }
        super.update(point, rcfPoint, timestamp, missing, forest);
    }

    // computes the normalization statistics
    protected Deviation[] getInitialDeviations() {
        Deviation[] tempList = new Deviation[NUMBER_OF_STATS * inputLength];
        for (int j = 0; j < NUMBER_OF_STATS * inputLength; j++) {
            tempList[j] = new Deviation(transformDecay);
        }
        for (int i = 0; i < initialValues.length; i++) {
            for (int j = 0; j < inputLength; j++) {
                tempList[j].update(initialValues[i][j]);
                double value = (i == 0) ? 0 : initialValues[i][j] - initialValues[i - 1][j];
                tempList[j + inputLength].update(value);
            }
        }
        for (int i = 0; i < initialValues.length; i++) {
            for (int j = 0; j < inputLength; j++) {
                tempList[j + 2 * inputLength].update(tempList[j].getDeviation());
                tempList[j + 3 * inputLength].update(tempList[j + inputLength].getMean());
                tempList[j + 4 * inputLength].update(tempList[j + inputLength].getDeviation());
            }
        }
        return tempList;
    }

    /**
     * a block which executes once; it first computes the multipliers for
     * normalization and then processes each of the stored inputs
     */

    protected void dischargeInitial(RandomCutForest forest) {
        Deviation tempTimeDeviation = new Deviation();
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            tempTimeDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double timeFactor = 1.0 + tempTimeDeviation.getDeviation();

        double quality = prepareInitialInput();
        Deviation[] deviations = getInitialDeviations();
        Arrays.fill(previousTimeStamps, initialTimeStamps[0]);

        for (int i = 0; i < valuesSeen; i++) {
            float[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], deviations, timeFactor);
            // missing values are null
            updateState(initialValues[i], scaledInput, initialTimeStamps[i], previousTimeStamps[shingleSize - 1], null);
            dataQuality[0].update(quality);
            if (forest != null) {
                if (forest.isInternalShinglingEnabled()) {
                    forest.update(scaledInput);
                } else {
                    forest.update(lastShingledPoint);
                }
            }
        }
        initialTimeStamps = null;
        initialValues = null;
    }
}
