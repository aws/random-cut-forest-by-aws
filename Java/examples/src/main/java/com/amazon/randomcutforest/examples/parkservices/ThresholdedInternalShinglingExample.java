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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;
import java.util.Random;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedInternalShinglingExample implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedInternalShinglingExample().run();
    }

    @Override
    public String command() {
        return "Thresholded_Multi_Dim_example";
    }

    @Override
    public String description() {
        return "Thresholded Multi Dimensional Example";
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

        long count = 0;
        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NONE;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .weightTime(0).transformMethod(transformMethod).normalizeTime(true).outputAfter(32)
                .initialAcceptFraction(0.125).build();
        ThresholdedRandomCutForest second = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01)
                .forestMode(ForestMode.TIME_AUGMENTED).weightTime(0).transformMethod(transformMethod)
                .normalizeTime(true).outputAfter(32).initialAcceptFraction(0.125).build();

        // ensuring that the parameters are the same; otherwise the grades/scores cannot
        // be the same
        // weighTime has to be 0
        forest.setLowerThreshold(1.1);
        second.setLowerThreshold(1.1);
        forest.setHorizon(0.75);
        second.setHorizon(0.75);

        long seed = new Random().nextLong();
        Random noise = new Random(0);

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                100, 5, seed, baseDimensions);

        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            // idea is that we expect the arrival order to be roughly 100 apart (say
            // seconds)
            // then the noise corresponds to a jitter; one can try TIME_AUGMENTED and
            // .normalizeTime(true)

            long timestamp = 100 * count + noise.nextInt(10) - 5;
            AnomalyDescriptor result = forest.process(point, timestamp);
            AnomalyDescriptor test = second.process(point, timestamp);
            checkArgument(Math.abs(result.getRcfScore() - test.getRcfScore()) < 1e-10, " error");
            checkArgument(Math.abs(result.getAnomalyGrade() - test.getAnomalyGrade()) < 1e-10, " error");

            if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                System.out
                        .println("timestamp " + count + " CHANGE " + Arrays.toString(dataWithKeys.changes[keyCounter]));
                ++keyCounter;
            }

            if (result.getAnomalyGrade() != 0) {
                System.out.print("timestamp " + count + " RESULT value " + result.getInternalTimeStamp() + " ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.getCurrentInput()[i] + ", ");
                }
                System.out.print("score " + result.getRcfScore() + ", grade " + result.getAnomalyGrade() + ", ");
                if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                    System.out.print(-result.getRelativeIndex() + " steps ago, ");
                }
                if (result.isExpectedValuesPresent()) {
                    if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                        System.out.print("instead of ");
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
                            if (result.getCurrentInput()[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print("( "
                                        + (result.getCurrentInput()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    }
                } else {
                    System.out.print("insufficient data to provide expected values");
                }
                System.out.println();
            }

            ++count;
        }

    }

}
