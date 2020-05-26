/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import java.util.stream.IntStream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import com.amazon.randomcutforest.RandomCutForest;

public class RandomCutForestSerDeTests {

    private RandomCutForestSerDe serializer = new RandomCutForestSerDe();

    @ParameterizedTest(name = "{index} => numDims={0}, numTrees={1}, numSamples={2}, numTrainSamples={3}, "
            + "numTestSamples={4}, enableParallel={5}, numThreads={6}")
    @CsvSource({ "1, 100, 256, 32, 1024, 0, 0", "1, 100, 256, 256, 1024, 0, 0", "1, 100, 256, 512, 1024, 0, 0",
            "1, 100, 256, 1024, 1024, 0, 0", "10, 100, 256, 32, 1024, 0, 0", "10, 100, 256, 256, 1024, 0, 0",
            "10, 100, 256, 512, 1024, 0, 0", "10, 100, 256, 1024, 1024, 0, 0", "1, 100, 256, 32, 1024, 1, 0",
            "1, 100, 256, 256, 1024, 1, 1", "1, 100, 256, 512, 1024, 1, 2", "1, 100, 256, 1024, 1024, 1, 4",
            "10, 100, 256, 32, 1024, 1, 0", "10, 100, 256, 256, 1024, 1, 1", "10, 100, 256, 512, 1024, 1, 2",
            "10, 100, 256, 1024, 1024, 1, 4", })
    public void toJsonString(int numDims, int numTrees, int numSamples, int numTrainSamples, int numTestSamples,
            int enableParallel, int numThreads) {
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0);
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

        String json = serializer.toJson(forest);
        RandomCutForest reForest = serializer.fromJson(json);

        double delta = Math.log(numSamples) / Math.log(2) * 0.05;
        for (double[] point : generate(numTestSamples, numDims)) {
            assertEquals(forest.getAnomalyScore(point), reForest.getAnomalyScore(point), delta);
            forest.update(point);
            reForest.update(point);
        }
    }

    private double[][] generate(int numSamples, int numDimensions) {
        return IntStream.range(0, numSamples).mapToObj(i -> new Random(0L).doubles(numDimensions).toArray())
                .toArray(double[][]::new);
    }
}
