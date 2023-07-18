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

package com.amazon.randomcutforest.examples.parkservices;

import java.util.Arrays;
import java.util.Random;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedImpute implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedImpute().run();
    }

    @Override
    public String command() {
        return "Thresholded_Imputation_example";
    }

    @Override
    public String description() {
        return "Thresholded Imputation Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;
        int baseDimensions = 1;

        long count = 0;

        int dropped = 0;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).imputationMethod(ImputationMethod.RCF)
                .forestMode(ForestMode.STREAMING_IMPUTE).transformMethod(TransformMethod.NORMALIZE_DIFFERENCE)
                .autoAdjust(true).build();

        long seed = new Random().nextLong();
        Random noisePRG = new Random(0);

        System.out.println("seed = " + seed);
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                100, 5, seed, baseDimensions);

        // as we loop over the data we will be dropping observations with probability
        // 0.2
        // note that as a result the predictor correct method would like be more
        // error-prone
        // note that estimation of the number of entries to be imputed is also another
        // estimation
        // therefore the overall method may have runaway effects if more values are
        // dropped.

        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            if (noisePRG.nextDouble() < 0.2 && !((keyCounter < dataWithKeys.changeIndices.length
                    && count == dataWithKeys.changeIndices[keyCounter]))) {
                dropped++;
                if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                    System.out.println(" dropped sequence " + (count) + " INPUT " + Arrays.toString(point) + " CHANGE "
                            + Arrays.toString(dataWithKeys.changes[keyCounter]));
                }
            } else {
                long newStamp = 100 * count + 2 * noisePRG.nextInt(10) - 5;
                AnomalyDescriptor result = forest.process(point, newStamp);

                if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                    System.out.println("sequence " + (count) + " INPUT " + Arrays.toString(point) + " CHANGE "
                            + Arrays.toString(dataWithKeys.changes[keyCounter]));
                    ++keyCounter;
                }

                if (result.getAnomalyGrade() != 0) {
                    System.out.print("sequence " + (count) + " RESULT value ");
                    for (int i = 0; i < baseDimensions; i++) {
                        System.out.print(result.getCurrentInput()[i] + ", ");
                    }
                    System.out.print("score " + result.getRCFScore() + ", grade " + result.getAnomalyGrade() + ", ");

                    if (result.isExpectedValuesPresent()) {
                        if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                            System.out.print(-result.getRelativeIndex() + " steps ago, instead of ");
                            for (int i = 0; i < baseDimensions; i++) {
                                System.out.print(result.getPastValues()[i] + ", ");
                            }
                            System.out.print("expected ");
                            for (int i = 0; i < baseDimensions; i++) {
                                System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                                if (result.getPastValues()[i] != result.getExpectedValuesList()[0][i]) {
                                    System.out.print(
                                            "( " + (result.getPastValues()[i] - result.getExpectedValuesList()[0][i])
                                                    + " ) ");
                                }
                            }
                        } else {
                            System.out.print("expected ");
                            for (int i = 0; i < baseDimensions; i++) {
                                System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                                if (result.getCurrentInput()[i] != result.getExpectedValuesList()[0][i]) {
                                    System.out.print(
                                            "( " + (result.getCurrentInput()[i] - result.getExpectedValuesList()[0][i])
                                                    + " ) ");
                                }
                            }
                        }
                    }
                    System.out.println();
                }
            }
            ++count;
        }
        System.out.println("Dropped " + dropped + " out of " + count);
    }

}
