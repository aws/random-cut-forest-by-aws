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

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.CorrectionMode;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedMultiDimensionalExample implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedMultiDimensionalExample().run();
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

        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 3;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                // dimensions is shingleSize x the number of base dimensions in input (in this
                // case 3)
                .dimensions(dimensions)
                // shingle size is the context (sliding) window of last contiguous observations
                .shingleSize(shingleSize)
                // fixed random seed would produce deterministic/reproducible results
                .randomSeed(0)
                // use about 50; more than 100 may not be useful
                .numberOfTrees(numberOfTrees)
                // samplesize should be large enough to cover the desired phenomenon; for a
                // 5-minute
                // interval reading if one is interested investigating anomalies over a weekly
                // pattern
                // there are 12 * 24 * 7 different
                // 5-minute intervals in a week. That being said, larger samplesize is a larger
                // model.
                .sampleSize(sampleSize)
                // shingling is now performed internally by default -- best not to change it
                // .internalShinglingEnabled(true)
                // change to different streaming transformations that are performed on the fly
                // note the transformation affects the characteristics of the anomaly that can
                // be
                // detected
                .transformMethod(TransformMethod.NORMALIZE)
                // the following would increase precision at the cost of recall
                // for the reverse, try ScoringStrategy.MULTI_MODE_RECALL
                // the default strategy is an attempted goldilocks version and may not work
                // for all data
                // .scoringStrategy(ScoringStrategy.MULTI_MODE)
                // the following will learn data (concept) drifts (also referered to as level
                // shifts) automatically and
                // stop repeated alarms. The reverse is also true -- to detect level shifts, set
                // the following to false
                // and test for continuous alarms
                .autoAdjust(true)
                // the following is a much coarser tool to eliminate repeated alarms
                // the descriptor below 'result' will contain information about different
                // correction/suppression modes
                // .alertOnce(true)
                .build();

        long seed = new Random().nextLong();
        System.out.println("seed = " + seed);

        // basic amplitude of the waves -- the parameter will be randomly scaled up
        // between 0-20 percent
        double amplitude = 100.0;

        // the amplitude of random noise it will be +ve/-ve uniformly at random
        double noise = 5.0;

        // the following controls the ratio of anomaly magnitude to noise
        // notice amplitude/noise would determine signal-to-noise ratio
        double anomalyFactor = 5;

        // the following determines if a random linear trend should be added
        boolean useSlope = false;

        // provide explanations and alternatives considered for non-anomalies
        boolean verboseSupression = true;

        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 24,
                amplitude, noise, seed, baseDimensions, anomalyFactor, useSlope);
        int keyCounter = 0;
        int count = 0;
        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor result = forest.process(point, 0L);

            if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                System.out.println(
                        "timestamp " + (count) + " CHANGE " + Arrays.toString(dataWithKeys.changes[keyCounter]));
                ++keyCounter;
            }

            if (result.getAnomalyGrade() != 0) {
                System.out.print("timestamp " + (count) + " RESULT value ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(point[i] + ", ");
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
                                System.out.print("( "
                                        + (result.getPastValues()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    } else {
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                            if (point[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print("( inferred change = "
                                        + (point[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    }
                }
                System.out.println();
            } else if (verboseSupression && result.getCorrectionMode() != CorrectionMode.NONE) {
                System.out.println(count + " corrected via " + result.getCorrectionMode().name());
            }
            ++count;
        }

    }

}
