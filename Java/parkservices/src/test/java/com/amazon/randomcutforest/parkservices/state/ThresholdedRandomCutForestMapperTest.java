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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class ThresholdedRandomCutForestMapperTest {

    @Test
    public void testRoundTripStandardShingleSizeOne() {
        int dimensions = 10;
        for (int trials = 0; trials < 1; trials++) {

            long seed = new Random().nextLong();
            RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions).randomSeed(seed);

            // note shingleSize == 1
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                    .randomSeed(seed).internalShinglingEnabled(true).anomalyRate(0.01).build();
            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                    .randomSeed(seed).anomalyRate(0.01).forestMode(ForestMode.STANDARD).internalShinglingEnabled(false)
                    .build();
            RandomCutForest forest = builder.build();

            Random r = new Random();
            for (int i = 0; i < 2000 + new Random().nextInt(1000); i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);

                assertEquals(firstResult.getDataConfidence(), secondResult.getDataConfidence(), 1e-10);
                assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
                assertEquals(firstResult.getRCFScore(), forest.getAnomalyScore(point), 1e-10);
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
                assertEquals(score, firstResult.getRCFScore(), 1e-10);
                assertEquals(score, secondResult.getRCFScore(), 1e-10);
                assertEquals(score, thirdResult.getRCFScore(), 1e-10);
                assertEquals(firstResult.getDataConfidence(), secondResult.getDataConfidence(), 1e-10);
                forest.update(point);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    public void testConversions(boolean internal) {
        int dimensions = 10;
        int shingleSize = 2;
        for (int trials = 0; trials < 5; trials++) {

            long seed = new Random().nextLong();
            System.out.println("Seed " + seed);
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).internalShinglingEnabled(internal)
                    .shingleSize(shingleSize).randomSeed(seed).build();

            // note shingleSize == 1
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                    .randomSeed(seed).internalShinglingEnabled(internal).shingleSize(shingleSize).anomalyRate(0.01)
                    .build();

            Random r = new Random(seed + 1);
            for (int i = 0; i < new Random(seed + 2).nextInt(1000); i++) {
                int length = (internal) ? dimensions / shingleSize : dimensions;
                double[] point = r.ints(length, 0, 50).asDoubleStream().toArray();
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
            for (int i = 0; i < new Random(seed + 3).nextInt(1000); i++) {
                int length = (internal) ? dimensions / shingleSize : dimensions;
                double[] point = r.ints(length, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);

                assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
                assertEquals(firstResult.getRCFScore(), forest.getAnomalyScore(point), 1e-10);
                forest.update(point);
            }

            // serialize + deserialize
            ThresholdedRandomCutForestMapper newMapper = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest third = newMapper.toModel(newMapper.toState(second));

            // update re-instantiated forest
            for (int i = 0; i < 100; i++) {
                int length = (internal) ? dimensions / shingleSize : dimensions;
                double[] point = r.ints(length, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);
                AnomalyDescriptor thirdResult = third.process(point, 0L);
                double score = forest.getAnomalyScore(point);
                assertEquals(score, firstResult.getRCFScore(), 1e-10);
                assertEquals(score, secondResult.getRCFScore(), 1e-10);
                assertEquals(score, thirdResult.getRCFScore(), 1e-10);
                assertEquals(firstResult.getDataConfidence(), thirdResult.getDataConfidence(), 1e-10);
                forest.update(point);
            }
        }
    }

    @Test
    public void testRoundTripStandardShingled() throws JsonProcessingException {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions).randomSeed(seed);

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).shingleSize(shingleSize).internalShinglingEnabled(false).anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).shingleSize(shingleSize).internalShinglingEnabled(false).anomalyRate(0.01).build();
        RandomCutForest forest = builder.build();

        // thresholds should not affect scores
        double value = 0.75 + 0.5 * new Random().nextDouble();
        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(10 * sampleSize, 50,
                shingleSize, baseDimensions, seed);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), forest.getAnomalyScore(point), 1e-4);
            forest.update(point);
        }

        ObjectMapper jsonMapper = new ObjectMapper();
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        String json = jsonMapper.writeValueAsString(mapper.toState(second));
        ThresholdedRandomCutForest third = mapper
                .toModel(jsonMapper.readValue(json, ThresholdedRandomCutForestState.class));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(100, 50, shingleSize,
                baseDimensions, seed);
        // update re-instantiated forest
        for (double[] point : testData.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);
            AnomalyDescriptor thirdResult = third.process(point, 0L);
            double score = forest.getAnomalyScore(point);
            assertEquals(score, firstResult.getRCFScore(), 1e-4);
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getDataConfidence(), thirdResult.getDataConfidence(), 1e-10);
            forest.update(point);
        }
    }

    @Test
    public void testRoundTripStandardShingledInternal() throws JsonProcessingException {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).internalShinglingEnabled(true)
                .shingleSize(shingleSize).randomSeed(seed).build();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .autoAdjust(true).boundingBoxCacheFraction(0).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .autoAdjust(true).build();

        double value = 0.75 + 0.5 * new Random().nextDouble();
        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        long count = 0;

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, count);
            AnomalyDescriptor secondResult = second.process(point, count);
            ++count;
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), forest.getAnomalyScore(point), 1e-4);
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
            }
            forest.update(point);
        }

        ObjectMapper jsonMapper = new ObjectMapper();
        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        String json = jsonMapper.writeValueAsString(mapper.toState(second));
        ThresholdedRandomCutForest third = mapper
                .toModel(jsonMapper.readValue(json, ThresholdedRandomCutForestState.class));

        MultiDimDataWithKey testData = ShingledMultiDimDataWithKeys.getMultiDimData(100, 50, 100, 5, seed,
                baseDimensions);

        // update re-instantiated forest
        for (double[] point : testData.data) {
            AnomalyDescriptor firstResult = first.process(point, count);
            AnomalyDescriptor secondResult = second.process(point, count);
            AnomalyDescriptor thirdResult = third.process(point, count);
            ++count;
            double score = forest.getAnomalyScore(point);
            assertEquals(score, firstResult.getRCFScore(), 1e-4);
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getDataConfidence(), thirdResult.getDataConfidence(), 1e-10);
            forest.update(point);
        }
        TimedRangeVector one = first.extrapolate(10);
        TimedRangeVector two = second.extrapolate(10);
        assertArrayEquals(one.upperTimeStamps, two.upperTimeStamps);
        assertArrayEquals(one.lowerTimeStamps, two.lowerTimeStamps);
        assertArrayEquals(one.timeStamps, two.timeStamps);
        assertArrayEquals(one.rangeVector.values, two.rangeVector.values, 1e-6f);
        assertArrayEquals(one.rangeVector.upper, two.rangeVector.upper, 1e-6f);
        assertArrayEquals(one.rangeVector.lower, two.rangeVector.lower, 1e-6f);
        for (int j = 0; j < 10; j++) {
            assert (one.lowerTimeStamps[j] <= one.timeStamps[j]);
            assert (one.upperTimeStamps[j] >= one.timeStamps[j]);
            assert (one.timeStamps[j] == count + j);
        }
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class)
    public void testRoundTripStandardInitial(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .autoAdjust(true).transformMethod(method).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .autoAdjust(true).transformMethod(method).build();

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(sampleSize, 50, 100, 5, seed,
                baseDimensions);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            second = mapper.toModel(mapper.toState(second));
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

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .transformMethod(method).autoAdjust(true).boundingBoxCacheFraction(0).weights(new double[] { 1.0 })
                .build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .transformMethod(method).autoAdjust(true).weights(new double[] { 1.0 }).build();

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
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
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
        }
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class, names = { "WEIGHTED", "NORMALIZE", "NORMALIZE_DIFFERENCE", "DIFFERENCE",
            "SUBTRACT_MA" })
    public void testRoundTripAugmentedInitial(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        double value = 0.75 + 0.25 * new Random().nextDouble();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(method).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0, 2.0 }).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(method).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0, 2.0 }).build();

        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(sampleSize, 50, 100, 5, seed,
                baseDimensions);

        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            second = mapper.toModel(mapper.toState(second));
        }

    }

    @Test
    public void testRoundTripAugmentedInitialNone() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        double value = 0.75 + 0.25 * new Random().nextDouble();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(TransformMethod.NONE).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0, 1.0 }).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(TransformMethod.NONE).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0, 1.0 }).build();

        first.setLowerThreshold(value);
        second.setLowerThreshold(value);

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(sampleSize, 50, 100, 5, seed,
                baseDimensions);

        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor firstResult = first.process(point, 0L);
            AnomalyDescriptor secondResult = second.process(point, 0L);

            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            second = mapper.toModel(mapper.toState(second));
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
        double value = 0.75 + 0.25 * new Random().nextDouble();

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(method).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0 }).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(method).anomalyRate(0.01).autoAdjust(true)
                .weights(new double[] { 1.0 }).build();

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
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            if (firstResult.getAnomalyGrade() > 0) {
                assertEquals(secondResult.getAnomalyGrade(), firstResult.getAnomalyGrade(), 1e-10);
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
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getAnomalyGrade(), thirdResult.getAnomalyGrade(), 1e-10);
            ++count;
        }
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class, names = { "WEIGHTED", "NORMALIZE", "NORMALIZE_DIFFERENCE", "DIFFERENCE",
            "SUBTRACT_MA" })
    public void testRoundTripTimeAugmentedNormalize(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(seed)
                .forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).transformMethod(method)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .weights(new double[] { 1.0, 2.0 }).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true)
                .internalShinglingEnabled(true).transformMethod(method).shingleSize(shingleSize).anomalyRate(0.01)
                .weights(new double[] { 1.0, 2.0 }).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            long stamp = 1000 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            ++count;
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
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
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
            ++count;
        }
    }

    @Test
    public void testRoundTripTimeAugmentedNone() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(seed)
                .forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).transformMethod(TransformMethod.NONE)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .weights(new double[] { 1.0, 1.0 }).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true)
                .internalShinglingEnabled(true).transformMethod(TransformMethod.NONE).shingleSize(shingleSize)
                .anomalyRate(0.01).weights(new double[] { 1.0, 1.0 }).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            long stamp = 1000 * count + r.nextInt(10) - 5;
            AnomalyDescriptor firstResult = first.process(point, stamp);
            AnomalyDescriptor secondResult = second.process(point, stamp);
            ++count;
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
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
            assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
            ++count;
        }
    }

    @ParameterizedTest
    @MethodSource("args")
    public void testRoundTripImputeInitial(TransformMethod transformMethod, ImputationMethod imputationMethod) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(transformMethod).imputationMethod(imputationMethod)
                .fillValues(new double[] { 1.0, 2.0 }).anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(transformMethod).imputationMethod(imputationMethod)
                .fillValues(new double[] { 1.0, 2.0 }).anomalyRate(0.01).build();

        Random r = new Random(0);
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(sampleSize, 50, 100, 5, seed,
                baseDimensions);

        for (double[] point : dataWithKeys.data) {
            if (r.nextDouble() > 0.1) {
                long stamp = 1000 * count + r.nextInt(10) - 5;
                AnomalyDescriptor firstResult = first.process(point, stamp);
                AnomalyDescriptor secondResult = second.process(point, stamp);
                assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            }
            ++count;

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            second = mapper.toModel(mapper.toState(second));
        }

    }

    @ParameterizedTest
    @MethodSource("args")
    public void testRoundTripImpute(TransformMethod transformMethod, ImputationMethod imputationMethod) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest first = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(seed)
                .forestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true).shingleSize(shingleSize)
                .transformMethod(transformMethod).imputationMethod(imputationMethod).fillValues(new double[] { 1.0 })
                .anomalyRate(0.01).build();
        ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                .randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true)
                .shingleSize(shingleSize).transformMethod(transformMethod).imputationMethod(imputationMethod)
                .fillValues(new double[] { 1.0 }).anomalyRate(0.01).build();

        Random r = new Random();
        long count = 0;
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(10 * sampleSize, 50, 100, 5,
                seed, baseDimensions);

        for (double[] point : dataWithKeys.data) {
            if (r.nextDouble() > 0.1) {
                long stamp = 1000 * count + r.nextInt(10) - 5;
                AnomalyDescriptor firstResult = first.process(point, stamp);
                AnomalyDescriptor secondResult = second.process(point, stamp);
                assertEquals(firstResult.getRCFScore(), secondResult.getRCFScore(), 1e-10);
            }
            ++count;
        }

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
            assertEquals(firstResult.getRCFScore(), thirdResult.getRCFScore(), 1e-10);
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
