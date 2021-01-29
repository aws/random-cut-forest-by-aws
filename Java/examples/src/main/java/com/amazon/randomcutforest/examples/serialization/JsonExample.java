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

package com.amazon.randomcutforest.examples.serialization;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Serialize a Random Cut Forest to JSON using
 * <a href="https://github.com/FasterXML/jackson">Jackson</a>.
 */
public class JsonExample implements Example {

    public static void main(String[] args) throws Exception {
        new JsonExample().run();
    }

    @Override
    public String command() {
        return "json";
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

        RandomCutForest forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();

        int dataSize = 4 * sampleSize;
        NormalMixtureTestData testData = new NormalMixtureTestData();
        for (double[] point : testData.generateTestData(dataSize, dimensions)) {
            forest.update(point);
        }

        // Convert to JSON and print the number of bytes

        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContext(true);
        mapper.setCopy(true);
        ObjectMapper jsonMapper = new ObjectMapper();

        String json = jsonMapper.writeValueAsString(mapper.toState(forest));

        System.out.printf("dimensions = %d, numberOfTrees = %d, sampleSize = %d, precision = %s%n", dimensions,
                numberOfTrees, sampleSize, precision);
        System.out.printf("JSON size = %d bytes%n", json.getBytes().length);

        // Restore from JSON and compare anomaly scores produced by the two forests

        RandomCutForest forest2 = mapper.toModel(jsonMapper.readValue(json, RandomCutForestState.class));

        int testSize = 100;
        double delta = Math.log(sampleSize) / Math.log(2) * 0.05;

        int differences = 0;
        int anomalies = 0;

        for (double[] point : testData.generateTestData(testSize, dimensions)) {
            double score = forest.getAnomalyScore(point);
            double score2 = forest2.getAnomalyScore(point);

            // we mostly care that points that are scored as an anomaly by one forest are
            // also scored as an anomaly by the other forest
            if (score > 1 || score2 > 1) {
                anomalies++;
                if (Math.abs(score - score2) > delta) {
                    differences++;
                }
            }

            forest.update(point);
            forest2.update(point);
        }

        // first validate that this was a nontrivial test
        if (anomalies == 0) {
            throw new IllegalStateException("test data did not produce any anomalies");
        }

        // validate that the two forests agree on anomaly scores
        if (differences >= 0.01 * testSize) {
            throw new IllegalStateException("restored forest does not agree with original forest");
        }

        System.out.println("Looks good!");
    }
}
