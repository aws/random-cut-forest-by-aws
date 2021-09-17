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

import java.util.Random;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class Thresholded1DGaussianMix implements Example {

    public static void main(String[] args) throws Exception {
        new Thresholded1DGaussianMix().run();
    }

    @Override
    public String command() {
        return "Thresholded_1D_Gaussian_example";
    }

    @Override
    public String description() {
        return "Thresholded one dimensional gassian mixture Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;

        int count = 0;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).forestMode(ForestMode.TIME_AUGMENTED)
                .build();

        long seed = new Random().nextLong();

        System.out.println("Anomalies would correspond to a run, based on a change of state.");
        System.out.println("Each change is normal <-> anomaly;  so after the second change the data is normal");
        System.out.println("seed = " + seed);
        NormalMixtureTestData normalMixtureTestData = new NormalMixtureTestData(10, 1.0, 50, 2.0, 0.01, 0.1);
        MultiDimDataWithKey dataWithKeys = normalMixtureTestData.generateTestDataWithKey(dataSize, 1, 0);

        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor result = forest.process(point, count);

            if (keyCounter < dataWithKeys.changeIndices.length
                    && result.getTimestamp() == dataWithKeys.changeIndices[keyCounter]) {
                System.out.println("timestamp " + (result.getTimestamp()) + " CHANGE");
                ++keyCounter;
            }

            if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                System.out.println("timestamp " + (count) + " CHANGE ");
                ++keyCounter;
            }

            if (result.getAnomalyGrade() != 0) {
                System.out.print("timestamp " + (count) + " RESULT value ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.getCurrentValues()[i] + ", ");
                }
                System.out.print("score " + result.getRcfScore() + ", grade " + result.getAnomalyGrade() + ", ");

                if (result.isExpectedValuesPresent()) {
                    if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                        System.out.print(-result.getRelativeIndex() + " steps ago, instead of ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getOldValues()[i] + ", ");
                        }
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                            if (result.getOldValues()[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print("( "
                                        + (result.getOldValues()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    } else {
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                            if (result.getCurrentValues()[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print(
                                        "( " + (result.getCurrentValues()[i] - result.getExpectedValuesList()[0][i])
                                                + " ) ");
                            }
                        }
                    }
                }
                System.out.println();
            }
            ++count;
        }
    }
}
