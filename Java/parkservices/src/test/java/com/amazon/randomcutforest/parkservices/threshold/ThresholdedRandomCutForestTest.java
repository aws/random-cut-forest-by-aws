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

package com.amazon.randomcutforest.parkservices.threshold;

import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
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
                () -> ThresholdedRandomCutForest.builder().compact(true).sampleSize(sampleSize).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());

        // have to enable internal shingling or keep it unspecified
        assertDoesNotThrow(
                () -> ThresholdedRandomCutForest.builder().compact(true).sampleSize(sampleSize).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED)
                        .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build());

        // imputefraction not allowed
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).sampleSize(sampleSize)
                        .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                        .setMode(ForestMode.TIME_AUGMENTED).useImputedFraction(0.5).internalShinglingEnabled(true)
                        .shingleSize(shingleSize).anomalyRate(0.01).build());

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).sampleSize(sampleSize)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).shingleSize(shingleSize).anomalyRate(0.01)
                .build();
        assertNotNull(forest.getInitialTimeStamps());
    }

    @Test
    public void testConfigAugmentTwo() {
        int baseDimensions = 2;
        int shingleSize = 1; // passes due to this
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(false).shingleSize(shingleSize)
                    .anomalyRate(0.01).build();
            assertEquals(forest.getForest().getDimensions(), dimensions + 1);

        });

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setMode(ForestMode.TIME_AUGMENTED).shingleSize(shingleSize).anomalyRate(0.01).build();
        assertTrue(forest.getForest().isInternalShinglingEnabled()); // default on

    }

    @Test
    public void testConfigImpute() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        // have to enable internal shingling or keep it unfixed
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());

        assertDoesNotThrow(() -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE)
                .shingleSize(shingleSize).anomalyRate(0.01).build());
    }

    @Test
    public void testConfigStandard() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        // have to enable internal shingling or keep it unfixed
        assertThrows(IllegalArgumentException.class,
                () -> ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STANDARD)
                        .useImputedFraction(0.5).internalShinglingEnabled(false).shingleSize(shingleSize)
                        .anomalyRate(0.01).build());

        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions).precision(Precision.FLOAT_32)
                    .randomSeed(seed).setMode(ForestMode.STANDARD).internalShinglingEnabled(false)
                    .shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STANDARD)
                    .shingleSize(shingleSize).anomalyRate(0.01).normalizeValues(true).startNormalization(111)
                    .stopNormalization(100).build();
        });
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).normalizeValues(true).startNormalization(111).stopNormalization(111).build();
        assertFalse(forest.getForest().isInternalShinglingEnabled()); // left to false
        assertEquals(forest.getInitialValues().length, 111);
        assertEquals(forest.getInitialTimeStamps().length, 111);
        assertEquals(forest.getStopNormalization(), 111);
        assertEquals(forest.getStartNormalization(), 111);
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

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.TIME_AUGMENTED).normalizeTime(true)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();
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
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            AnomalyDescriptor thirdResult = third.process(point, stamp);
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            ++count;
        }
    }

    @Test
    void testImputeConfig() {
        int baseDimensions = 1;
        int shingleSize = 2;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        // not providing values
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.FIXED_VALUES).normalizeTime(true)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        // incorrect number of values to fill
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.FIXED_VALUES)
                    .fillValues(new double[] { 0.0, 17.0 }).normalizeTime(true).internalShinglingEnabled(true)
                    .shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        // normalization undefined for fill with 0's
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.ZERO).fillValues(new double[] { 2.0 })
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).normalizeValues(true)
                    .build();
        });

        // normalization unclear for Fixed
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.FIXED_VALUES)
                    .fillValues(new double[] { 2.0 }).internalShinglingEnabled(true).shingleSize(shingleSize)
                    .anomalyRate(0.01).normalizeValues(true).build();
        });

        // normalization of time not useful for impute
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.FIXED_VALUES)
                    .fillValues(new double[] { 2.0 }).internalShinglingEnabled(true).shingleSize(shingleSize)
                    .anomalyRate(0.01).normalizeTime(true).build();
        });

        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(ImputationMethod.FIXED_VALUES)
                    .fillValues(new double[] { 2.0 }).internalShinglingEnabled(true).shingleSize(shingleSize)
                    .anomalyRate(0.01).build();
        });
    }

    @ParameterizedTest
    @EnumSource(ImputationMethod.class)
    void testImpute(ImputationMethod method) {
        int baseDimensions = 1;
        int shingleSize = 1;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        // shingle size 1 ie not useful for impute
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(method).normalizeTime(true)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        int newShingleSize = 4;
        int newDimensions = baseDimensions * newShingleSize;

        // time is used in impute and not for normalization
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(newDimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .setMode(ForestMode.STREAMING_IMPUTE).fillIn(method).normalizeTime(true)
                    .internalShinglingEnabled(true).shingleSize(newShingleSize).anomalyRate(0.01).build();
        });

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(newDimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE).fillIn(method)
                .internalShinglingEnabled(true).shingleSize(newShingleSize).anomalyRate(0.01).useImputedFraction(0.76)
                .fillValues(new double[] { 0 }).build();

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random(0);
        int count = 0;
        for (int i = 0; i < 200 + new Random().nextInt(100); i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        // note every will have an update
        assertEquals(forest.getForest().getTotalUpdates(), count);
        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        assert (result.getAnomalyGrade() > 0);
        assert (result.isExpectedValuesPresent());
        assert (result.getRelativeIndex() == 0);
        assertArrayEquals(result.getExpectedValuesList()[0], fixedData, 1e-6);
        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one arise from the actual input
        assertEquals(forest.getForest().getTotalUpdates(), count + 4);
        // triggerring consecutive anomalies (no differencing) -- this should fail
        // because we discover previous point is still the cause
        assertEquals(forest.process(newData, (long) count * 113 + 1113).getAnomalyGrade(), 0);
        assert (forest.process(new double[] { 20 }, (long) count * 113 + 1226).getAnomalyGrade() > 0);

        long stamp = (long) count * 113 + 1226;
        // time has to increase
        assertThrows(IllegalArgumentException.class, () -> {
            forest.process(new double[] { 20 }, stamp);
        });
    }

    // streaming normalization does not seem to make sense with fill-in with 0 or
    // fixed values (in actual data)
    @ParameterizedTest
    @EnumSource(value = ImputationMethod.class, names = { "PREVIOUS", "RCF" })
    void testImputeNormalize(ImputationMethod method) {
        int baseDimensions = 1;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE).fillIn(method)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).useImputedFraction(0.76)
                .fillValues(new double[] { 1.0 }).outputAfter(10).normalizeValues(true).build();

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random();
        long count = 0;
        int spread = (int) Math.floor(random.nextDouble() * 100);
        for (int i = 0; i < 200 + spread; i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        // note every will have an update
        assertEquals(forest.getForest().getTotalUpdates(), count);
        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        assert (result.getAnomalyGrade() > 0);
        assert (result.isExpectedValuesPresent());
        // relative index may not be 0 because of normalization having changed the
        // values
        if (method == RCF || method == PREVIOUS) {
            assertArrayEquals(result.getExpectedValuesList()[0], fixedData, 1e-3);
        } else {
            // possible errors
            assertEquals(result.getExpectedValuesList()[0][0], fixedData[0], fixedData[0] / 2);
        }
        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one arise from the actual input
        assertEquals(forest.getForest().getTotalUpdates(), count + 4);
        // triggerring consecutive anomalies -- this should fail
        assert (forest.process(newData, (long) count * 113 + 1113).getAnomalyGrade() == 0);
    }

    @ParameterizedTest
    @EnumSource(ImputationMethod.class)
    void testImputeDifference(ImputationMethod method) {
        int baseDimensions = 1;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE).fillIn(method)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).useImputedFraction(0.76)
                .fillValues(new double[] { 1.0 }).differencing(true).build();

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random();
        int count = 0;
        for (int i = 0; i < 200 + new Random().nextInt(100); i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        // note every will have an update
        assertEquals(forest.getForest().getTotalUpdates(), count);
        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        assert (result.getAnomalyGrade() > 0);
        assert (result.isExpectedValuesPresent());
        // the other impute methods generate too much noise
        if (method == RCF || method == PREVIOUS) {
            assertArrayEquals(result.getExpectedValuesList()[0], fixedData, 1e-3);
        }

        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one arise from the actual input
        assertEquals(forest.getForest().getTotalUpdates(), count + 4);
        // triggerring consecutive anomalies (but differencing) -- this should fail
        assert (forest.process(newData, (long) count * 113 + 1113).getAnomalyGrade() == 0);
    }

    // streaming normalization does not seem to make sense with fill-in with 0 or
    // fixed values (in actual data)
    @ParameterizedTest
    @EnumSource(value = ImputationMethod.class, names = { "PREVIOUS", "RCF" })
    void testImputeDifferenceNormalized(ImputationMethod method) {
        int baseDimensions = 1;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setMode(ForestMode.STREAMING_IMPUTE).fillIn(method)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).useImputedFraction(0.76)
                .fillValues(new double[] { 1.0 }).differencing(true).normalizeValues(true).build();

        forest.setZfactor(3.0);

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random();
        int count = 0;
        for (int i = 0; i < 200 + new Random().nextInt(100); i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        // note every will have an update
        assertEquals(forest.getForest().getTotalUpdates(), count);
        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        assert (result.getAnomalyGrade() > 0);
        assert (result.isExpectedValuesPresent());
        // the other impute methods generate too much noise
        if (method == RCF || method == PREVIOUS) {
            assertArrayEquals(result.getExpectedValuesList()[0], fixedData, 1e-3);
        }

        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one arise from the actual input
        assertEquals(forest.getForest().getTotalUpdates(), count + 4);
        // triggerring consecutive anomalies -- but streaming normalization is in effect
        // with differencing!
        assert (forest.process(newData, (long) count * 113 + 1113).getAnomalyGrade() > 0);
    }

}
