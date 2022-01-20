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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

@Tag("functional")
public class RCF3Test {

    @Test
    public void basicTest() {
        int numberOfTrees = 30; // a test on a single tree -- there will always be floating point issues
                                // from adding many different scores
        int shingleSize = 8;
        int numberOfAttributes = 5;
        int dimensions = shingleSize * numberOfAttributes;
        long randomSeed = new Random().nextLong();
        System.out.println("seed " + randomSeed);
        int DATA_SIZE = 2000;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0,
                numberOfAttributes);

        RCF3 forest = RCF3.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                .boundingBoxCacheFraction(0.8).internalShinglingEnabled(true).randomSeed(randomSeed).outputAfter(1)
                .initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(256).build();
        RandomCutForest oldForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions)
                .shingleSize(shingleSize).boundingBoxCacheFraction(1.0).internalShinglingEnabled(true)
                .randomSeed(randomSeed).outputAfter(1).initialAcceptFraction(0.1).precision(Precision.FLOAT_32)
                .sampleSize(256).build();

        double score = 0;
        int count = 0;
        for (int j = 0; j < dataWithKey.data.length; j++) {
            double oldVal = oldForest.getAnomalyScore(dataWithKey.data[j]);
            double val = forest.dynamicScore(dataWithKey.data[j]);
            double newVal = forest.getAnomalyScore(dataWithKey.data[j]);

            if (Math.abs(val - oldVal) > 1e-5 || Math.abs(val - newVal) > 1e-5) {
                ++count;
            }
            score += val;
            forest.update(dataWithKey.data[j]);
            oldForest.update(dataWithKey.data[j]);

        }

        System.out.println(count);
        assert (count < 0.001 * dataWithKey.data.length);
        System.out.println(count + " values differ; average score " + score / dataWithKey.data.length);
    }

    @Test
    public void dynamicScoreTest() {
        int numberOfTrees = 30;
        int shingleSize = 8;
        int numberOfAttributes = 5;
        int dimensions = shingleSize * numberOfAttributes;

        int DATA_SIZE = 100000;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0,
                numberOfAttributes);
        double[][] shingledData = generateShingledData(dataWithKey.data, shingleSize, numberOfAttributes, false);

        assertEquals(shingledData.length, dataWithKey.data.length - shingleSize + 1);

        RCF3 forest = RCF3.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                .boundingBoxCacheFraction(1).internalShinglingEnabled(true).randomSeed(200).outputAfter(1)
                .initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(256).build();
        RandomCutForest otherForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions)
                .shingleSize(shingleSize).boundingBoxCacheFraction(0.1).randomSeed(200).outputAfter(1)
                .initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(256).build();

        int count = 0;
        for (int j = 0; j < shingleSize - 1; j++) {
            forest.update(dataWithKey.data[j]);
            ++count;
        }

        double val = 0;
        for (int j = 0; j < shingledData.length; j++) {

            double score = forest.getAnomalyScore(dataWithKey.data[j + count]);
            double newScore = forest.dynamicScore(dataWithKey.data[count + j]);
            double otherScore = otherForest.getAnomalyScore(shingledData[j]);
            assertEquals(score, newScore, 0.1);
            assertEquals(score, otherScore, 1e-5);

            forest.update(dataWithKey.data[j + count]);
            otherForest.update(shingledData[j]);
        }
    }

    @Test
    public void extrapolateTest() {
        int numberOfTrees = 30;
        int shingleSize = 8;
        int numberOfAttributes = 5;
        int dimensions = shingleSize * numberOfAttributes;

        int DATA_SIZE = 10000;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0,
                numberOfAttributes);

        RCF3 forest = RCF3.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                .boundingBoxCacheFraction(1).internalShinglingEnabled(true).randomSeed(200).outputAfter(1)
                .initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(256).build();
        RandomCutForest otherForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions)
                .shingleSize(shingleSize).boundingBoxCacheFraction(1).internalShinglingEnabled(true).randomSeed(200)
                .outputAfter(1).initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(256).build();

        int test = 100;

        for (int j = 0; j < dataWithKey.data.length - test; j++) {
            forest.update(dataWithKey.data[j]);
            otherForest.update(dataWithKey.data[j]);
        }

        for (int j = dataWithKey.data.length - test; j < dataWithKey.data.length; j++) {
            double[] first = forest.extrapolate(1);
            double[] second = otherForest.extrapolate(1);
            assertArrayEquals(first, second, 1e-2);
            forest.update(dataWithKey.data[j]);
            otherForest.update(dataWithKey.data[j]);
        }

    }

    @Test
    public void timeTest() {
        int numberOfTrees = 30;
        int shingleSize = 8;
        int numberOfAttributes = 5;
        int dimensions = shingleSize * numberOfAttributes;

        int DATA_SIZE = 100000;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0,
                numberOfAttributes);

        RCF3 forest = RCF3.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                .boundingBoxCacheFraction(1.0).internalShinglingEnabled(true).randomSeed(200).outputAfter(1)
                // .parallelExecutionEnabled(true)
                .initialAcceptFraction(0.1).precision(Precision.FLOAT_32).sampleSize(500).build();

        double score = 0;
        for (int j = 0; j < dataWithKey.data.length; j++) {
            double val = forest.dynamicScore(dataWithKey.data[j]);
            score += val;
            forest.update(dataWithKey.data[j]);
        }

        System.out.println(score / dataWithKey.data.length);
    }
}