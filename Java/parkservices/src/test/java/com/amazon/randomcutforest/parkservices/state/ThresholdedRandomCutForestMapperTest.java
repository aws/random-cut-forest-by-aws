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

package com.amazon.randomcutforest.parkservices.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedRandomCutForestMapperTest {

    @Test
    public void testRoundTripStandardShingleSizeOne() {
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
                    .setForestMode(ForestMode.STANDARD).internalShinglingEnabled(false).build();
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
    public void testConversions() {
        int dimensions = 10;
        for (int trials = 0; trials < 10; trials++) {

            long seed = new Random().nextLong();
            RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).internalShinglingEnabled(false).randomSeed(seed).build();

            // note shingleSize == 1
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).anomalyRate(0.01).build();

            Random r = new Random();
            for (int i = 0; i < new Random().nextInt(1000); i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                first.process(point, 0L);
                forest.update(point);
            }

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContextEnabled(true);
            mapper.setSaveTreeStateEnabled(true);
            mapper.setPartialTreeStateEnabled(true);
            RandomCutForest copyForest = mapper.toModel(mapper.toState(forest));

            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest(copyForest, 0.01, null);
            //
            for (int i = 0; i < new Random().nextInt(1000); i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);

                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-10);
                forest.update(point);
            }

            // serialize + deserialize
            ThresholdedRandomCutForestMapper newMapper = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest third = newMapper.toModel(newMapper.toState(second));

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

        // thresholds should not affect scores
        double value = 0.75 + 0.5 * new Random().nextDouble();
        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

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
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).internalShinglingEnabled(true).shingleSize(shingleSize).randomSeed(seed)
                .build();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).adjustThreshold(true).boundingBoxCacheFraction(0).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).adjustThreshold(true).build();

        double value = 0.75 + 0.5 * new Random().nextDouble();
        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        Random r = new Random();
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-4);
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
                assert (firstResult.getRcfScore() >= value);
            }
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

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class)
    public void testRoundTripStandard(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = 0;
        new Random().nextLong();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).transformMethod(method).adjustThreshold(true)
                .boundingBoxCacheFraction(0).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).anomalyRate(0.01).transformMethod(method).adjustThreshold(true).build();

        double value = 0.75 + 0.5 * new Random().nextDouble();
        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        Random r = new Random();
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
                assert (firstResult.getRcfScore() >= value);
            }

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
            assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
        }
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class)
    public void testRoundTripTimeAugmented(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        double value = 1.0 + 0.25 * new Random().nextDouble();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setForestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true).shingleSize(shingleSize)
                .transformMethod(method).anomalyRate(0.01).adjustThreshold(true).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setForestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true).shingleSize(shingleSize)
                .transformMethod(method).anomalyRate(0.01).adjustThreshold(true).build();

        first.setLowerThreshold(value);
        second.setLowerThreshold(value);
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
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
                assert (firstResult.getRcfScore() >= value);
            }
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
            assertEquals(firstResult.getAnomalyGrade(), thirdResult.getAnomalyGrade(), 1e-10);
            ++count;
        }
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class)
    public void testRoundTripTimeAugmentedNormalize(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setForestMode(ForestMode.TIME_AUGMENTED)
                .normalizeTime(true).transformMethod(method).internalShinglingEnabled(true).shingleSize(shingleSize)
                .anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setForestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).internalShinglingEnabled(true)
                .transformMethod(method).shingleSize(shingleSize).anomalyRate(0.01).build();

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

    @ParameterizedTest
    @MethodSource("args")
    public void testRoundTripImputeDifference(TransformMethod transformMethod, ImputationMethod imputationMethod) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).setForestMode(ForestMode.STREAMING_IMPUTE)
                .internalShinglingEnabled(true).shingleSize(shingleSize).transformMethod(TransformMethod.NONE)
                .fillIn(imputationMethod).fillValues(new double[] { 1.0 }).anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .setForestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true).shingleSize(shingleSize)
                .transformMethod(TransformMethod.NONE).fillIn(imputationMethod).fillValues(new double[] { 1.0 })
                .anomalyRate(0.01).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            if (r.nextDouble() > 0.1) {
                long stamp = 1000 * count + r.nextInt(10) - 5;
                AnomalyDescriptor firstResult = first.process(point, stamp);
                AnomalyDescriptor secondResult = second.process(point, stamp);
                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            }
            ++count;
        }
        ;

        assertThrows(IllegalArgumentException.class, () -> first.process(dataWithKeys.data[0], 0));
        // serialize + deserialize
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.getMultiDimData(100, 50, 100, 5, seed,
                baseDimensions);

        // update re-instantiated forest
        for (double[] point : testData.data) {
            long stamp = 1000 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, stamp);
            // AnomalyDescriptor secondResult = second.process(point, stamp);
            AnomalyDescriptor thirdResult = third.process(point, stamp);
            // assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
            assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
            ++count;
        }
    }

    static Stream<Arguments> args() {
        return transformMethodStream().flatMap(
                classParameter -> imputationMethod().map(testParameter -> Arguments.of(classParameter, testParameter)));
    }

    static Stream<ImputationMethod> imputationMethod() {
        return Stream.of(ImputationMethod.values());
    }

    static Stream<TransformMethod> transformMethodStream() {
        return Stream.of(TransformMethod.values());
    }
}
