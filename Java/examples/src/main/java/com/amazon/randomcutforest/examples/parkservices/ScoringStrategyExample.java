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

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ScoringStrategyExample implements Example {

    public static void main(String[] args) throws Exception {
        new ScoringStrategyExample().run();
    }

    @Override
    public String command() {
        return "Scoring_strategy_example";
    }

    @Override
    public String description() {
        return "Scoring Strategy Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;

        long seed = new Random().nextLong();
        long count = 0;
        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NORMALIZE;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(seed).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).scoringStrategy(ScoringStrategy.EXPECTED_INVERSE_DEPTH)
                .transformMethod(transformMethod).outputAfter(32).initialAcceptFraction(0.125).build();
        ThresholdedRandomCutForest second = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(seed).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).scoringStrategy(ScoringStrategy.MULTI_MODE)
                .transformMethod(transformMethod).outputAfter(32).initialAcceptFraction(0.125).build();
        ThresholdedRandomCutForest third = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(seed).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).scoringStrategy(ScoringStrategy.MULTI_MODE_RECALL)
                .transformMethod(transformMethod).outputAfter(32).initialAcceptFraction(0.125).build();

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                100, 5, seed, baseDimensions);

        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor result = forest.process(point, 0L);
            AnomalyDescriptor multi_mode = second.process(point, 0L);
            AnomalyDescriptor multi_mode_recall = third.process(point, 0L);

            checkArgument(Math.abs(result.getRCFScore() - multi_mode.getRCFScore()) < 1e-10, " error");
            checkArgument(Math.abs(result.getRCFScore() - multi_mode_recall.getRCFScore()) < 1e-10, " error");

            if (keyCounter < dataWithKeys.changeIndices.length && count == dataWithKeys.changeIndices[keyCounter]) {
                System.out
                        .println("timestamp " + count + " CHANGE " + Arrays.toString(dataWithKeys.changes[keyCounter]));
                ++keyCounter;
            }

            printResult("MULTI_MODE_RECALL", multi_mode_recall, count, baseDimensions);
            printResult("EXPECTED_INVERSE_DEPTH", result, count, baseDimensions);
            printResult("MULTI_MODE", multi_mode, count, baseDimensions);
            ++count;
        }
    }

    void printResult(String description, AnomalyDescriptor result, long count, int baseDimensions) {
        if (result.getAnomalyGrade() != 0) {
            System.out.print(description + " timestamp " + count + " RESULT value ");
            for (int i = 0; i < baseDimensions; i++) {
                System.out.print(result.getCurrentInput()[i] + ", ");
            }
            System.out.print("score " + result.getRCFScore() + ", grade " + result.getAnomalyGrade() + ", ");
            if (result.getRelativeIndex() != 0) {
                System.out.print(-result.getRelativeIndex() + " steps ago, ");
            }
            if (result.isExpectedValuesPresent()) {
                if (result.getRelativeIndex() != 0) {
                    System.out.print("instead of ");
                    for (int i = 0; i < baseDimensions; i++) {
                        System.out.print(result.getPastValues()[i] + ", ");
                    }
                    System.out.print("expected ");
                    for (int i = 0; i < baseDimensions; i++) {
                        System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                        if (result.getPastValues()[i] != result.getExpectedValuesList()[0][i]) {
                            System.out.print(
                                    "( " + (result.getPastValues()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                        }
                    }
                } else {
                    System.out.print("expected ");
                    for (int i = 0; i < baseDimensions; i++) {
                        System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                        if (result.getCurrentInput()[i] != result.getExpectedValuesList()[0][i]) {
                            System.out.print("( " + (result.getCurrentInput()[i] - result.getExpectedValuesList()[0][i])
                                    + " ) ");
                        }
                    }
                }
            } else {
                System.out.print("insufficient data to provide expected values");
            }
            System.out.println();
        }
    }

}
