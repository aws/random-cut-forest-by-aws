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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

/**
 * This class validates that forests configured with different execution modes
 * (sequential or parallel) or different internal data representations are
 * executing the algorithm steps in the same way.
 */
@Tag("functional")
public class RandomCutForestConsistencyFunctionalTest {

    private int dimensions = 5;
    private int sampleSize = 128;
    private long randomSeed = 123L;
    private int testSize = 2048;

    @Test
    public void testConsistentScoring() {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions).sampleSize(sampleSize)
                .randomSeed(randomSeed);

        RandomCutForest pointerCachedSequential = builder.compact(false).boundingBoxCacheFraction(1.0)
                .parallelExecutionEnabled(false).build();
        RandomCutForest pointerCachedParallel = builder.compact(false).boundingBoxCacheFraction(1.0)
                .parallelExecutionEnabled(true).build();
        RandomCutForest pointerCachedRandomSequential = builder.compact(false)
                .boundingBoxCacheFraction(new Random().nextDouble()).parallelExecutionEnabled(false).build();
        RandomCutForest pointerCachedRandomParallel = builder.compact(false)
                .boundingBoxCacheFraction(new Random().nextDouble()).parallelExecutionEnabled(true).build();
        RandomCutForest pointerUncachedSequential = builder.compact(false).boundingBoxCacheFraction(0.0)
                .parallelExecutionEnabled(false).build();
        RandomCutForest pointerUncachedParallel = builder.compact(false).boundingBoxCacheFraction(0.0)
                .parallelExecutionEnabled(true).build();
        RandomCutForest compactCachedSequential = builder.compact(true).boundingBoxCacheFraction(1.0)
                .parallelExecutionEnabled(false).build();
        RandomCutForest compactCachedParallel = builder.compact(true).boundingBoxCacheFraction(1.0)
                .parallelExecutionEnabled(true).build();
        RandomCutForest compactUncachedSequential = builder.compact(true).boundingBoxCacheFraction(0.0)
                .parallelExecutionEnabled(false).build();
        RandomCutForest compactUncachedParallel = builder.compact(true).boundingBoxCacheFraction(0.0)
                .parallelExecutionEnabled(true).build();
        RandomCutForest compactCachedRandomSequential = builder.compact(true)
                .boundingBoxCacheFraction(new Random().nextDouble()).parallelExecutionEnabled(false).build();
        RandomCutForest compactCachedRandomParallel = builder.compact(true)
                .boundingBoxCacheFraction(new Random().nextDouble()).parallelExecutionEnabled(true).build();

        NormalMixtureTestData testData = new NormalMixtureTestData();
        double delta = 1e-10;
        int anomalies = 0;

        for (double[] point : testData.generateTestData(testSize, dimensions, 99)) {
            double score = pointerCachedSequential.getAnomalyScore(point);

            if (score > 0) {
                anomalies++;
            }

            assertEquals(score, pointerCachedParallel.getAnomalyScore(point), delta);
            assertEquals(score, pointerUncachedSequential.getAnomalyScore(point), delta);
            assertEquals(score, pointerUncachedParallel.getAnomalyScore(point), delta);
            assertEquals(score, compactCachedSequential.getAnomalyScore(point), delta);
            assertEquals(score, compactCachedParallel.getAnomalyScore(point), delta);
            assertEquals(score, compactUncachedSequential.getAnomalyScore(point), delta);
            assertEquals(score, compactUncachedParallel.getAnomalyScore(point), delta);
            assertEquals(score, pointerCachedRandomSequential.getAnomalyScore(point), delta);
            assertEquals(score, pointerCachedRandomParallel.getAnomalyScore(point), delta);
            assertEquals(score, compactCachedRandomSequential.getAnomalyScore(point), delta);
            assertEquals(score, compactCachedRandomParallel.getAnomalyScore(point), delta);

            pointerCachedSequential.update(point);
            pointerCachedParallel.update(point);
            pointerUncachedSequential.update(point);
            pointerUncachedParallel.update(point);
            pointerCachedRandomSequential.update(point);
            pointerCachedRandomParallel.update(point);
            compactCachedSequential.update(point);
            compactCachedParallel.update(point);
            compactUncachedSequential.update(point);
            compactUncachedParallel.update(point);
            compactCachedRandomSequential.update(point);
            compactCachedRandomParallel.update(point);
        }

        // verify that the test is nontrivial
        assertTrue(anomalies > 0);
    }

    @Test
    public void testConsistentScoringSinglePrecision() {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions).sampleSize(sampleSize)
                .randomSeed(randomSeed).parallelExecutionEnabled(false).compact(true);

        RandomCutForest compactFloatCached = builder.boundingBoxCacheFraction(1.0).precision(Precision.FLOAT_32)
                .build();
        RandomCutForest compactFloatCachedParallel = builder.boundingBoxCacheFraction(1.0).precision(Precision.FLOAT_32)
                .parallelExecutionEnabled(true).build();
        RandomCutForest compactFloatUncached = builder.boundingBoxCacheFraction(0.0).precision(Precision.FLOAT_32)
                .build();
        RandomCutForest compactFloatCachedRandom = builder.boundingBoxCacheFraction(new Random().nextDouble())
                .precision(Precision.FLOAT_32).build();
        RandomCutForest compactFloatCachedRandomParallel = builder.boundingBoxCacheFraction(new Random().nextDouble())
                .precision(Precision.FLOAT_32).parallelExecutionEnabled(true).build();
        RandomCutForest compactFloatUncachedParallel = builder.boundingBoxCacheFraction(0.0)
                .precision(Precision.FLOAT_32).parallelExecutionEnabled(true).build();
        RandomCutForest compactDoubleCached = builder.boundingBoxCacheFraction(1.0).precision(Precision.FLOAT_64)
                .build();

        NormalMixtureTestData testData = new NormalMixtureTestData();
        int anomalies = 0;

        for (double[] point : testData.generateTestData(testSize, dimensions, 99)) {
            double score = compactFloatCached.getAnomalyScore(point);

            if (score > 0) {
                anomalies++;
            }

            assertEquals(score, compactFloatUncached.getAnomalyScore(point), 1e-10);
            assertEquals(score, compactFloatUncachedParallel.getAnomalyScore(point), 1e-10);
            assertEquals(score, compactFloatCachedRandom.getAnomalyScore(point), 1e-10);
            assertEquals(score, compactFloatCachedRandomParallel.getAnomalyScore(point), 1e-10);

            // we expect some loss of precision when comparing to the score computed as a
            // double
            assertEquals(score, compactDoubleCached.getAnomalyScore(point), 1e-2);

            compactFloatCached.update(point);
            compactFloatCachedParallel.update(point);
            compactFloatUncached.update(point);
            compactFloatUncachedParallel.update(point);
            compactFloatCachedRandom.update(point);
            compactFloatCachedRandomParallel.update(point);
            compactDoubleCached.update(point);
        }

        // verify that the test is nontrivial
        assertTrue(anomalies > 0);
    }

}
