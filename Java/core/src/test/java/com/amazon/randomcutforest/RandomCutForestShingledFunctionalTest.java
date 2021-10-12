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

import static com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys.generateShingledData;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
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

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    public void InternalShinglingTest(boolean rotation) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 2;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        int numTrials = 1; // test is exact equality, reducing the number of trials
        int length = 4000 * sampleSize;

        for (int i = 0; i < numTrials; i++) {

            RandomCutForest first = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                    .internalRotationEnabled(rotation).shingleSize(shingleSize).build();

            RandomCutForest second = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(false)
                    .shingleSize(shingleSize).build();

            RandomCutForest third = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(false).shingleSize(1)
                    .build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    seed + i, baseDimensions);

            double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, rotation);

            assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

            int count = shingleSize - 1;
            // insert initial points
            for (int j = 0; j < shingleSize - 1; j++) {
                first.update(dataWithKeys.data[j]);
            }

            for (int j = 0; j < shingledData.length; j++) {
                // validate equality of points
                for (int y = 0; y < baseDimensions; y++) {
                    int position = (rotation) ? (count % shingleSize) : shingleSize - 1;
                    assertEquals(dataWithKeys.data[count][y], shingledData[j][position * baseDimensions + y], 1e-10);
                }

                double firstResult = first.getAnomalyScore(dataWithKeys.data[count]);
                first.update(dataWithKeys.data[count]);
                ++count;
                double secondResult = second.getAnomalyScore(shingledData[j]);
                second.update(shingledData[j]);
                double thirdResult = third.getAnomalyScore(shingledData[j]);
                third.update(shingledData[j]);

                assertEquals(firstResult, secondResult, 1e-10);
                assertEquals(secondResult, thirdResult, 1e-10);
            }
            PointStoreFloat store = (PointStoreFloat) first.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
            store = (PointStoreFloat) second.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
            store = (PointStoreFloat) third.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
        }
    }
}
