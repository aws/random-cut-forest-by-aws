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

package com.amazon.randomcutforest.serialize;

import com.amazon.randomcutforest.ForestState;
import com.amazon.randomcutforest.RandomCutForest;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class RandomCutForestSerDeTests {

    private RandomCutForestSerDe serializer = new RandomCutForestSerDe();

    @ParameterizedTest(name = "{index} => numDims={0}, numTrees={1}, numSamples={2}, numTrainSamples={3}, "
            + "numTestSamples={4}, enableParallel={5}, numThreads={6}")

    @CsvSource({ "1, 100, 256, 32, 1024, 0, 0", "1, 100, 256, 256, 1024, 0, 0", "1, 100, 256, 512, 1024, 0, 0",
            "1, 100, 256, 1024, 1024, 0, 0", "10, 100, 256, 32, 1024, 0, 0", "10, 100, 256, 256, 1024, 0, 0",
            "10, 100, 256, 512, 1024, 0, 0", "10, 100, 256, 1024, 1024, 0, 0", "1, 100, 256, 32, 1024, 1, 0",
            "1, 100, 256, 256, 1024, 1, 1", "1, 100, 256, 512, 1024, 1, 2", "1, 100, 256, 1024, 1024, 1, 4",
            "10, 100, 256, 32, 1024, 1, 0", "10, 100, 256, 1024, 1024, 1, 1", "4, 10, 256, 1024, 1024, 1, 2",
            "10, 100, 256, 1024, 10240, 1, 6" })

    public void toJsonString(int numDims, int numTrees, int numSamples, int numTrainSamples, int numTestSamples,
            int enableParallel, int numThreads) {
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).compactEnabled(true).saveTreeData(true);
        if (enableParallel == 0) {
            forestBuilder.parallelExecutionEnabled(false);
        }
        if (numThreads > 0) {
            forestBuilder.threadPoolSize(numThreads);
        }
        RandomCutForest forest = forestBuilder.build();

        for (double[] point : generate(numTrainSamples, numDims)) {
            forest.update(point);
        }

        ForestState forestState = forest.getForestState();
        String json = serializer.toJson(forestState);

        ForestState reForestState = serializer.fromJson(json);
        System.out.println(" Size " + json.length());

        RandomCutForest reForest = RandomCutForest.getForest(reForestState);

        int num = 0;
        int numForDimOne = 0;
        double delta = Math.log(numSamples) / Math.log(2) * 0.05;
        for (double[] point : generate(numTestSamples, numDims)) {
            double score = forest.getAnomalyScore(point);
            double newScore = reForest.getAnomalyScore(point);
            if (numDims>1) {
                assertTrue(Math.abs(score - newScore) < delta);
                if ((score>1) && (Math.abs( score - newScore) > 0.05 * score))
                    num++;
            } else {
                if (((score>1) || (newScore>1)) && (Math.abs(score - newScore) > delta))
                    numForDimOne++;
            }

            forest.update(point);
            reForest.update(point);
        }
        /**
         * It may be the case that more than epsilon = 0.05 fraction of the points are not within
         * 5% of the score,  but then all those scores are lower than 1. Note that for numDims>1
         * the difference never exceeds delta.
         *
         * For numDims == 1, there may be more noise; and hence the test corresponds to
         */
        if (numDims>1) {
            assertTrue(num < 0.05 * numTestSamples);
        } else {
            assertTrue(numForDimOne < 0.01 * numTestSamples);
        }

    }

    @ParameterizedTest(name = "{index} => numDims={0}, numTrees={1}, numSamples={2}, numTrainSamples={3}, "
            + "numTestSamples={4}, enableParallel={5}, numThreads={6}")
    @CsvSource({ "10, 100, 256, 2048, 100, 0, 0" })
    public void timeTest(int numDims, int numTrees, int numSamples, int numTrainSamples, int numTestSamples,
            int enableParallel, int numThreads) {
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).compactEnabled(true).saveTreeData(false);
        if (enableParallel == 0) {
            forestBuilder.parallelExecutionEnabled(false);
        }
        if (numThreads > 0) {
            forestBuilder.threadPoolSize(numThreads);
        }
        RandomCutForest forest = forestBuilder.build();

        for (double[] point : generate(numTrainSamples, numDims)) {
            forest.update(point);
        }
        ForestState forestState = forest.getForestState();
        String json = serializer.toJson(forestState);
        double delta = Math.log(numSamples) / Math.log(2) * 0.05;
        System.out.println("Size " + json.length());
        ForestState reForestState = serializer.fromJson(json);

        for (int i = 0; i < numTestSamples; i++) {
            reForestState = serializer.fromJson(json);
            RandomCutForest reForest = RandomCutForest.getForest(reForestState);
            double[] point = generate(1, numDims)[0];
            assertTrue(Math.abs(forest.getAnomalyScore(point) - reForest.getAnomalyScore(point)) < delta);
            reForest.update(point);
            forest.update(point);
            reForestState = reForest.getForestState();
            json = serializer.toJson(reForestState);
        }
    }

    private double[][] generate(int numSamples, int numDimensions) {
        return IntStream.range(0, numSamples).mapToObj(i -> new Random(i).doubles(numDimensions).toArray())
                .toArray(double[][]::new);
    }
}
