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

package com.amazon.randomcutforest.tree;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.RandomCutForestTest;
import com.amazon.randomcutforest.config.Precision;

public class BoxCacheTest {

    @Test
    public void testChangingBoundingBoxFloat32() {
        int dimensions = 4;
        int numberOfTrees = 1;
        int sampleSize = 64;
        int dataSize = 1000 * sampleSize;
        Random random = new Random();
        long seed = random.nextLong();
        double[][] big = RandomCutForestTest.generateShingledData(dataSize, dimensions, 2);
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(Precision.FLOAT_32).randomSeed(seed)
                .boundingBoxCacheFraction(0).build();
        RandomCutForest otherForest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(Precision.FLOAT_32).randomSeed(seed)
                .boundingBoxCacheFraction(1).build();
        int num = 0;
        for (double[] point : big) {
            ++num;
            if (num % sampleSize == 0) {
                forest.setBoundingBoxCacheFraction(random.nextDouble());
            }
            assertEquals(forest.getAnomalyScore(point), otherForest.getAnomalyScore(point));
            forest.update(point);
            otherForest.update(point);
        }
    }

    @Test
    public void testChangingBoundingBoxFloat64() {
        int dimensions = 10;
        int numberOfTrees = 1;
        int sampleSize = 256;
        int dataSize = 400 * sampleSize;
        Random random = new Random();
        double[][] big = RandomCutForestTest.generateShingledData(dataSize, dimensions, 2);
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(Precision.FLOAT_64)
                .randomSeed(random.nextLong()).boundingBoxCacheFraction(random.nextDouble()).build();

        for (double[] point : big) {
            forest.setBoundingBoxCacheFraction(random.nextDouble());
            forest.update(point);
        }
    }

}
