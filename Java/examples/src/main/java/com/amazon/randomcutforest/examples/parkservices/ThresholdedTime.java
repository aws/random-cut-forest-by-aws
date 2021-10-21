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

public class ThresholdedTime implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedTime().run();
    }

    @Override
    public String command() {
        return "Thresholded_Time_example";
    }

    @Override
    public String description() {
        return "Thresholded Time Example";
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
                .sampleSize(sampleSize).internalShinglingEnabled(true).precision(precision).anomalyRate(0.01)
                .forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).build();

        long seed = new Random().nextLong();

        double[] data = new double[] { 1.0 };

        System.out.println("seed = " + seed);
        NormalMixtureTestData normalMixtureTestData = new NormalMixtureTestData(10, 50);
        MultiDimDataWithKey dataWithKeys = normalMixtureTestData.generateTestDataWithKey(dataSize, 1, 0);

        /**
         * the anomalies will move from normal -> anomalous -> normal starts from normal
         */
        boolean anomalyState = false;

        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            long time = (long) (1000L * count + Math.floor(10 * point[0]));
            AnomalyDescriptor result = forest.process(data, time);

            if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                System.out.print("Sequence " + count + " stamp " + (result.getInternalTimeStamp()) + " CHANGE ");
                if (!anomalyState) {
                    System.out.println(" to Distribution 1 ");
                } else {
                    System.out.println(" to Distribution 0 ");
                }
                anomalyState = !anomalyState;
                ++keyCounter;
            }

            if (result.getAnomalyGrade() != 0) {
                System.out.print("Sequence " + count + " stamp " + (result.getInternalTimeStamp()) + " RESULT ");
                System.out.print("score " + result.getRCFScore() + ", grade " + result.getAnomalyGrade() + ", ");

                if (result.isExpectedValuesPresent()) {
                    if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                        System.out.print(-result.getRelativeIndex() + " steps ago, instead of stamp "
                                + result.getPastTimeStamp());
                        System.out.print(", expected timestamp " + result.getExpectedTimeStamp() + " ( "
                                + (result.getPastTimeStamp() - result.getExpectedTimeStamp() + ")"));
                    } else {
                        System.out.print("expected " + result.getExpectedTimeStamp() + " ( "
                                + (result.getInternalTimeStamp() - result.getExpectedTimeStamp() + ")"));
                    }
                }
                System.out.println();
            }
            ++count;
        }

    }

}
