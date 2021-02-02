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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class DynamicSampling implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicSampling().run();
    }

    @Override
    public String command() {
        return "dynamic sampling";
    }

    @Override
    public String description() {
        return "check dynamic sampling";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.DOUBLE;
        int dataSize = 4 * sampleSize;
        NormalMixtureTestData testData = new NormalMixtureTestData();

        RandomCutForest forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();
        RandomCutForest forest2 = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();

        int first_anomalies = 0;
        int second_anomalies = 0;
        forest2.setLambda(10 * forest2.getLambda());

        for (double[] point : testData.generateTestData(dataSize, dimensions)) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            forest.update(point);
            forest2.update(point);
        }
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies);
        // should be roughly equal

        first_anomalies = second_anomalies = 0;
        testData = new NormalMixtureTestData(-3);
        for (double[] point : testData.generateTestData(dataSize, dimensions)) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            forest.update(point);
            forest2.update(point);
        }
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies);
        // forest2 should adapt faster

        first_anomalies = second_anomalies = 0;
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContext(true);
        mapper.setCopy(true);
        RandomCutForest copyForest = mapper.toModel(mapper.toState(forest));
        copyForest.setLambda(10 * forest.getLambda());
        // force an adjustment to catch up
        testData = new NormalMixtureTestData(-10);
        int forced_change_anomalies = 0;
        for (double[] point : testData.generateTestData(dataSize, dimensions)) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            if (copyForest.getAnomalyScore(point) > 1.0) {
                forced_change_anomalies++;
            }
            copyForest.update(point);
            forest.update(point);
            forest2.update(point);
        }
        // both should show the similar rate of adjustment
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies
                + ", forced (first) " + forced_change_anomalies);

    }
}
