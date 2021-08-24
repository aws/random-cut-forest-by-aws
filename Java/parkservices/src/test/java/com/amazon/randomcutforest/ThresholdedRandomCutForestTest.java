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

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForestMapper;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedRandomCutForestTest {

    @Test
    public void testConfigAugmentOne() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigAugmentTwo() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 1; // passes due to this
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertDoesNotThrow(() -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED)
                .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigAugmentThree() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertDoesNotThrow(() -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigAugmentFour() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigAugmentFive() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertDoesNotThrow(() -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE)
                .shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigAugmentSix() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 1; // due to this
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE)
                        .shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testRoundTripStandard() {
        int dimensions = 10;
        for (int trials = 0; trials < 10; trials++) {

            long seed = new Random().nextLong();
            RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed);

            // note shingleSize == 1
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(true).anomalyRate(0.01).build();
            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).anomalyRate(0.01)
                    .setMode(ForestMode.STANDARD).internalShinglingEnabled(false).build();
            RandomCutForest forest = builder.build();

            Random r = new Random();
            for (int i = 0; i < new Random().nextInt(1000); i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);

                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-10);
                forest.update(point);
            }

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

            // update re-instantiated forest
            for (int i = 0; i < 100; i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);
                AnomalyDescriptor thirdResult = third.process(point, 0L);
                double score = forest.getAnomalyScore(point);
                assertEquals(score, firstResult.getRcfScore(), 1e-10);
                assertEquals(score, secondResult.getRcfScore(), 1e-10);
                assertEquals(score, thirdResult.getRcfScore(), 1e-10);
                forest.update(point);
            }
        }
    }

    @Test
    public void testRoundTripStandardShingled() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed);

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).shingleSize(shingleSize)
                .anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).shingleSize(shingleSize)
                .anomalyRate(0.01).build();
        RandomCutForest forest = builder.build();

        Random r = new Random();
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(10 * sampleSize, 50,
                shingleSize, baseDimensions, seed);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-4);
            forest.update(point);
        }

        // serialize + deserialize
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(100, 50, shingleSize,
                baseDimensions, seed);
        // update re-instantiated forest
        for (double[] point : testData.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);
            AnomalyDescriptor thirdResult = third.process(point, 0L);
            double score = forest.getAnomalyScore(point);
            assertEquals(score, firstResult.getRcfScore(), 1e-4);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            forest.update(point);
        }
    }

    @Test
    public void testRoundTripStandardShingledInternal() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).internalShinglingEnabled(true).shingleSize(shingleSize).randomSeed(seed);

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).build();
        RandomCutForest forest = builder.build();

        Random r = new Random();
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-4);
            forest.update(point);
        }

        // serialize + deserialize
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.getMultiDimData(100, 50, 100, 5, seed,
                baseDimensions);

        // update re-instantiated forest
        for (double[] point : testData.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);
            AnomalyDescriptor thirdResult = third.process(point, 0L);
            double score = forest.getAnomalyScore(point);
            assertEquals(score, firstResult.getRcfScore(), 1e-4);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            forest.update(point);
        }
    }

    @Test
    public void testRoundTripTimeAugmented() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true).shingleSize(shingleSize)
                .anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true).shingleSize(shingleSize)
                .anomalyRate(0.01).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            long stamp = 100 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            ++count;
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
        }

        // serialize + deserialize
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.getMultiDimData(100, 50, 100, 5, seed,
                baseDimensions);

        // update re-instantiated forest
        for (double[] point : testData.data) {
            long stamp = 100 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);
            AnomalyDescriptor thirdResult = third.process(point, 0L);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            ++count;
        }
    }

    @Test
    public void testRoundTripTimeAugmentedNormalize() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).internalShinglingEnabled(true).shingleSize(shingleSize).randomSeed(seed)
                .build();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            long stamp = 1000 * count + r.nextInt(10) - 5;
            double score = forest.getAnomalyScore(point);
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            ++count;
            assertEquals(firstResult.getRcfScore(), score, 0.2);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            forest.update(point);
        }

        // serialize + deserialize
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.getMultiDimData(100, 50, 100, 5, seed,
                baseDimensions);

        // update re-instantiated forest
        for (double[] point : testData.data) {
            long stamp = 100 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            AnomalyDescriptor thirdResult = third.process(point, stamp);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            forest.update(point);
            ++count;
        }
    }
}
