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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;

public class RandomCutForestSerDeTests {

    private static final int dimensions = 4;
    private static final int numberOfTrees = 30;
    private static final int sampleSize = 28000;
    private static double[][] testData;

    @BeforeAll
    public static void oneTimeSetUp() {
        testData = new double[4 * sampleSize][dimensions];
        for (int i = 0; i < 4 * sampleSize; i++) {
            for (int j = 0; j < dimensions; j++) {
                testData[i][j] = Math.random();
            }
        }
    }

    private static void initForest(RandomCutForest forest) {
        for (double[] point : testData) {
            forest.update(point);
        }
    }

    public static Stream<RandomCutForest> forestProvider() {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).compactEnabled(true);

        RandomCutForest pointerForest = builder.compactEnabled(false).build();
        initForest(pointerForest);
        RandomCutForest compactDouble = builder.compactEnabled(true).precision(Precision.DOUBLE).build();
        initForest(compactDouble);
        RandomCutForest compactSingle = builder.compactEnabled(true).precision(Precision.SINGLE).build();
        initForest(compactSingle);

        return Stream.of(pointerForest, compactDouble, compactSingle);
    }

    private RandomCutForestSerDe serializer;

    @BeforeEach
    public void setUp() {
        serializer = new RandomCutForestSerDe();
        serializer.getMapper().setSaveExecutorContext(true);
    }

    @ParameterizedTest
    @MethodSource("forestProvider")
    public void testRoundTrip(RandomCutForest forest) {
        String json = serializer.toJson(forest);
        RandomCutForest forest2 = serializer.fromJson(json);

        int numTestSamples = 100;
        double delta = Math.log(sampleSize) / Math.log(2) * 0.05;

        int differences = 0;
        int anomalies = 0;

        for (int i = 0; i < numTestSamples; i++) {
            double[] point = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                point[j] = Math.random();
            }

            double score = forest.getAnomalyScore(point);
            double score2 = forest2.getAnomalyScore(point);

            if (score > 1 || score2 > 1) {
                anomalies++;
                if (Math.abs(score - score2) > delta) {
                    differences++;
                }
            }

            forest.update(point);
            forest2.update(point);
        }

        assertTrue(anomalies > 0);
        assertTrue(differences < 0.01 * numTestSamples);
    }
}
