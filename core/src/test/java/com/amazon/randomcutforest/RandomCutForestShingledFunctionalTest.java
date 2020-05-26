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

package com.amazon.randomcutforest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.ShingleBuilder;

@Tag("functional")
public class RandomCutForestShingledFunctionalTest {
    private static int numberOfTrees;
    private static int sampleSize;
    private static int dimensions;
    private static int randomSeed;
    private static int shingleSize;
    private static ShingleBuilder shingleBuilder;
    private static RandomCutForest forest;

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    @BeforeAll
    public static void oneTimeSetUp() {
        numberOfTrees = 100;
        sampleSize = 256;
        dimensions = 2;
        randomSeed = 123;
        shingleSize = 3;

        shingleBuilder = new ShingleBuilder(dimensions, shingleSize);

        forest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shingleBuilder.getShingledPointSize()).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        dataSize = 10_000;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 5.0;
        anomalySigma = 1.5;
        transitionToAnomalyProbability = 0.01;
        transitionToBaseProbability = 0.4;

        NormalMixtureTestData generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, dimensions);

        for (int i = 0; i < dataSize; i++) {
            shingleBuilder.addPoint(data[i]);
            if (shingleBuilder.isFull()) {
                forest.update(shingleBuilder.getShingle());
            }
        }
    }

    @Test
    public void testExtrapolateBasic() {
        double[] result = forest.extrapolateBasic(shingleBuilder.getShingle(), 4, dimensions, false);
        assertEquals(4 * dimensions, result.length);

        result = forest.extrapolateBasic(shingleBuilder.getShingle(), 4, dimensions, true, 2);
        assertEquals(4 * dimensions, result.length);

        result = forest.extrapolateBasic(shingleBuilder, 4);
        assertEquals(4 * dimensions, result.length);

        // use a block size which is too big
        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(shingleBuilder.getShingle(), 4, 4, true, 2));
    }
}
