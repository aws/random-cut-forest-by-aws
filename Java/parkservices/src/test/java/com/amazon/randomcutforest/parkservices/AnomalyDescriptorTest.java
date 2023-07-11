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

package com.amazon.randomcutforest.parkservices;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class AnomalyDescriptorTest {

    @ParameterizedTest
    @EnumSource(ScoringStrategy.class)
    public void PastValuesTest(ScoringStrategy strategy) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 10; // just once since testing exact equality
        int length = 40 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int outputAfter = 2 + 1;
            int shingleSize = 1 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(rng.nextLong())
                    .outputAfter(outputAfter).scoringStrategy(strategy).internalShinglingEnabled(true)
                    .shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            int count = 0;
            for (double[] point : dataWithKeys.data) {
                if (count == 82) {
                    point[0] += 10000; // introducing an anomaly
                }
                AnomalyDescriptor firstResult = first.process(point, 0L);
                assertArrayEquals(firstResult.getCurrentInput(), point, 1e-6);
                assertEquals(firstResult.scoringStrategy, strategy);
                if (count < outputAfter || count < shingleSize) {
                    assertEquals(firstResult.getRCFScore(), 0);
                } else {
                    // distances can be 0
                    assertTrue(strategy == ScoringStrategy.DISTANCE || firstResult.getRCFScore() > 0);
                    assertTrue(strategy == ScoringStrategy.DISTANCE || firstResult.threshold > 0);
                    assertEquals(firstResult.getScale().length, baseDimensions);
                    assertEquals(firstResult.getShift().length, baseDimensions);
                    assertTrue(firstResult.getRelativeIndex() <= 0);
                    if (count == 82 && strategy != ScoringStrategy.DISTANCE) {
                        // because distances are 0 till sampleSize; by which time
                        // forecasts would be reasonable
                        assertTrue(firstResult.getAnomalyGrade() > 0);
                    }
                    if (firstResult.getAnomalyGrade() > 0) {
                        assertNotNull(firstResult.getPastValues());
                        assertEquals(firstResult.getPastValues().length, baseDimensions);
                        if (firstResult.getRelativeIndex() == 0) {
                            assertArrayEquals(firstResult.getPastValues(), firstResult.getCurrentInput(), 1e-10);
                        }

                        assertNotNull(firstResult.getRelevantAttribution());
                        assertEquals(firstResult.getRelevantAttribution().length, baseDimensions);
                        assertEquals(firstResult.attribution.getHighLowSum(), firstResult.getRCFScore(), 1e-6);
                        // the reverse of this condition need not be true -- the predictor corrector
                        // often may declare grade 0 even when score is greater than threshold, to
                        // account for shingling and initial results that populate the thresholder
                        assertTrue(strategy == ScoringStrategy.MULTI_MODE_RECALL
                                || firstResult.getRCFScore() >= firstResult.getThreshold());
                    } else {
                        assertTrue(firstResult.getRelativeIndex() == 0);
                    }
                }
                ++count;

            }
        }
    }

    @ParameterizedTest
    @EnumSource(ScoringStrategy.class)
    public void TimeAugmentedTest(ScoringStrategy strategy) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 10; // just once since testing exact equality
        int length = 40 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int outputAfter = 2 + 1;
            int shingleSize = 1 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(rng.nextLong())
                    .outputAfter(outputAfter).forestMode(ForestMode.TIME_AUGMENTED).scoringStrategy(strategy)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            int count = 0;
            for (double[] point : dataWithKeys.data) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                assertArrayEquals(firstResult.getCurrentInput(), point, 1e-6);
                assertEquals(firstResult.scoringStrategy, strategy);
                if (count < outputAfter || count < shingleSize) {
                    assertEquals(firstResult.getRCFScore(), 0);
                } else {
                    // distances can be 0
                    assertTrue(strategy == ScoringStrategy.DISTANCE || firstResult.getRCFScore() > 0);
                    assertTrue(strategy == ScoringStrategy.DISTANCE || firstResult.threshold > 0);
                    assertEquals(firstResult.getScale().length, baseDimensions + 1);
                    assertEquals(firstResult.getShift().length, baseDimensions + 1);
                    assertTrue(firstResult.getRelativeIndex() <= 0);
                    if (firstResult.getAnomalyGrade() > 0) {
                        assertNotNull(firstResult.getPastValues());
                        assertEquals(firstResult.getPastValues().length, baseDimensions);
                        if (firstResult.getRelativeIndex() == 0) {
                            assertArrayEquals(firstResult.getPastValues(), firstResult.getCurrentInput(), 1e-10);
                        }
                        assertEquals(firstResult.attribution.getHighLowSum(), firstResult.getRCFScore(), 1e-6);
                        assertNotNull(firstResult.getRelevantAttribution());
                        assertEquals(firstResult.getRelevantAttribution().length, baseDimensions);
                        assertTrue(strategy == ScoringStrategy.MULTI_MODE_RECALL
                                || firstResult.getRCFScore() >= firstResult.getThreshold());
                    }
                }
                ++count;

            }
        }
    }
}
