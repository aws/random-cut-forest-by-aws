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

package com.amazon.randomcutforest.examples.dynamicconfiguration;

import java.time.Duration;
import java.time.Instant;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class DynamicThroughput implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicThroughput().run();
    }

    @Override
    public String command() {
        return "dynamic caching";
    }

    @Override
    public String description() {
        return "serialize a Random Cut Forest as a JSON string";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.DOUBLE;
        int dataSize = 10 * sampleSize;
        NormalMixtureTestData testData = new NormalMixtureTestData();
        // generate data once to eliminate caching issues
        testData.generateTestData(dataSize, dimensions);
        testData.generateTestData(sampleSize, dimensions);

        for (int i = 0; i < 5; i++) {

            RandomCutForest forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions).randomSeed(0)
                    .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();
            RandomCutForest forest2 = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions)
                    .randomSeed(0).numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();
            forest2.setBoundingBoxCacheFraction(i * 0.25);

            int anomalies = 0;

            for (double[] point : testData.generateTestData(dataSize, dimensions)) {
                double score = forest.getAnomalyScore(point);
                double score2 = forest2.getAnomalyScore(point);

                if (Math.abs(score - score2) > 1e-10) {
                    anomalies++;
                }
                forest.update(point);
                forest2.update(point);
            }

            Instant start = Instant.now();

            for (double[] point : testData.generateTestData(sampleSize, dimensions)) {
                double score = forest.getAnomalyScore(point);
                double score2 = forest2.getAnomalyScore(point);

                if (Math.abs(score - score2) > 1e-10) {
                    anomalies++;
                }
                forest.update(point);
                forest2.update(point);
            }

            Instant finish = Instant.now();

            // first validate that this was a nontrivial test
            if (anomalies > 0) {
                throw new IllegalStateException("score mismatch");
            }

            System.out.println("So far so good! Caching fraction = " + (i * 0.25) + ", Time ="
                    + Duration.between(start, finish).toMillis() + " ms (note only one forest is changing)");
        }

    }

}
