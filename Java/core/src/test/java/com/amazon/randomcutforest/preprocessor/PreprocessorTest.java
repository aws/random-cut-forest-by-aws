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

package com.amazon.randomcutforest.preprocessor;

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.config.ForestMode.STANDARD;
import static com.amazon.randomcutforest.config.ForestMode.STREAMING_IMPUTE;
import static com.amazon.randomcutforest.config.ForestMode.TIME_AUGMENTED;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.LINEAR;
import static com.amazon.randomcutforest.config.ImputationMethod.NEXT;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.config.TransformMethod.NONE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE_DIFFERENCE;
import static com.amazon.randomcutforest.preprocessor.Preprocessor.copyAtEnd;
import static java.lang.Math.abs;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.statistics.Deviation;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class PreprocessorTest {

    @Test
    void testConfig() {
        assertThrows(IllegalArgumentException.class, () -> copyAtEnd(new double[2], new double[3]));
        assertThrows(IllegalArgumentException.class, () -> copyAtEnd(new float[4], new float[5]));
        assertNull(Preprocessor.copyIfNotnull((float[]) null));
        assertNull(Preprocessor.copyIfNotnull((double[]) null));
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(null).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(null).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).inputLength(10).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .forestMode(STANDARD).inputLength(10).dimensions(12).build());
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).inputLength(12).dimensions(12)
                    .build();
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .forestMode(STANDARD).inputLength(12).dimensions(12).initialShingledInput(new double[1]).build());
        assertDoesNotThrow(() -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).inputLength(12)
                .dimensions(12).initialShingledInput(new double[12]).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).inputLength(6)
                        .dimensions(12).shingleSize(2).initialShingledInput(new double[6]).build());

        assertDoesNotThrow(() -> new Preprocessor.Builder<>().transformMethod(NONE).forestMode(STANDARD).inputLength(6)
                .dimensions(12).shingleSize(2).initialShingledInput(new double[12]).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NORMALIZE)
                .forestMode(STANDARD).inputLength(12).dimensions(12).startNormalization(0).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NORMALIZE_DIFFERENCE).forestMode(STANDARD)
                        .inputLength(12).dimensions(12).startNormalization(0).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .forestMode(TIME_AUGMENTED).inputLength(12).dimensions(12).build());
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(TIME_AUGMENTED).inputLength(12).dimensions(13)
                    .build();
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(TIME_AUGMENTED).inputLength(12).dimensions(13).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(TIME_AUGMENTED).inputLength(12).dimensions(14).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(STREAMING_IMPUTE).inputLength(12).dimensions(12).shingleSize(1).build());
        assertDoesNotThrow(() -> new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2)
                .forestMode(TIME_AUGMENTED).inputLength(6).dimensions(14).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(TIME_AUGMENTED)
                        .inputLength(6).dimensions(14).initialShingledInput(new double[14]).build());
        assertDoesNotThrow(() -> new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2)
                .forestMode(TIME_AUGMENTED).inputLength(6).dimensions(14).initialShingledInput(new double[12]).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(TIME_AUGMENTED)
                        .inputLength(6).initialPoint(new float[12]).dimensions(14).build());
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(TIME_AUGMENTED).inputLength(6)
                    .dimensions(14).initialPoint(new float[14]).build();
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(-2).forestMode(TIME_AUGMENTED).inputLength(6).dimensions(14).build());

        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).normalizeTime(true)
                    .forestMode(TIME_AUGMENTED).inputLength(6).dimensions(14).build();
        });

        // external shingling in STANDARD mode
        assertDoesNotThrow(() -> {
            IPreprocessor preprocessor = new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2)
                    .forestMode(TIME_AUGMENTED).inputLength(6).dimensions(14).build();
            // need a forest
            assertThrows(IllegalArgumentException.class,
                    () -> preprocessor.getScaledShingledInput(new double[6], 0L, null, null));
        });

        // internal shingling
        assertDoesNotThrow(() -> {
            IPreprocessor preprocessor = new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2)
                    .forestMode(STANDARD).inputLength(6).dimensions(12).build();
            assertDoesNotThrow(() -> preprocessor.getScaledShingledInput(new double[6], 0L, null, null));
            assertNull(preprocessor.invertInPlaceRecentSummaryBlock(null));
            SampleSummary summary = new SampleSummary(6);
            summary.summaryPoints = new float[1][6];
            summary.measure = new float[1][2];
            assertThrows(IllegalArgumentException.class, () -> preprocessor.invertInPlaceRecentSummaryBlock(summary));
            assertThrows(IllegalArgumentException.class, () -> preprocessor.setDefaultFill(new double[7]));
            assertDoesNotThrow(() -> preprocessor.setDefaultFill(new double[6]));
            assertThrows(IllegalArgumentException.class,
                    () -> ((Preprocessor) preprocessor).setPreviousTimeStamps(new long[5]));
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(STANDARD).weights(new double[1]).inputLength(6).dimensions(12).build());

        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(STANDARD).weights(new double[2]).inputLength(6).dimensions(12).build());

        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(STANDARD)
                        .weights(new double[] { 1.0, 1.0 }).inputLength(6).dimensions(12).build());

        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .forestMode(STANDARD).inputLength(6).dimensions(12).build());
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).normalizeTime(true).forestMode(STANDARD)
                    .inputLength(6).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .shingleSize(2).forestMode(STANDARD).inputLength(5).dimensions(12).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .inputLength(5).dimensions(5).startNormalization(0).normalizeTime(true).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().transformMethod(NONE)
                .inputLength(1).dimensions(1).weights(new double[] { 0.5 }).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(1)
                .startNormalization(0).transformMethod(NORMALIZE_DIFFERENCE).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(1)
                .startNormalization(0).transformMethod(NORMALIZE_DIFFERENCE).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(1)
                .forestMode(STREAMING_IMPUTE).imputationMethod(FIXED_VALUES).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(2)
                .forestMode(STREAMING_IMPUTE).imputationMethod(FIXED_VALUES).shingleSize(2).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Preprocessor.Builder<>().inputLength(1).dimensions(2).forestMode(STREAMING_IMPUTE)
                        .imputationMethod(FIXED_VALUES).shingleSize(2).fillValues(new double[2]).build());
        assertDoesNotThrow(() -> new Preprocessor.Builder<>().inputLength(1).dimensions(2).forestMode(STREAMING_IMPUTE)
                .imputationMethod(FIXED_VALUES).shingleSize(2).fillValues(new double[1]).build());
    }

    public void preprocessorPlusForest(int seed, ForestMode mode, TransformMethod method, ImputationMethod imputeMethod,
            boolean internalShinglingHint, int shingleSize) {
        int dataSize = 1000;
        int sampleSize = 256;
        int tempDimensions = (mode == TIME_AUGMENTED) ? 2 * shingleSize : shingleSize;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, seed, 1,
                false);
        Preprocessor.Builder<?> builder = Preprocessor.builder().inputLength(1).dimensions(tempDimensions)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).transformMethod(method).randomSeed(seed + 1)
                .forestMode(mode);
        if (mode == STREAMING_IMPUTE) {
            builder.imputationMethod(imputeMethod);
            builder.fastForward(new Random().nextDouble() < 0.5);
            if (imputeMethod == FIXED_VALUES) {
                builder.fillValues(new double[] { 5 });
            }
        }
        if (imputeMethod != null) {
            builder.imputationMethod(imputeMethod);
        }
        boolean internal = ((internalShinglingHint || method != NONE) && mode != STREAMING_IMPUTE);
        Preprocessor preprocessor = builder.build(); // polymorphism
        RandomCutForest forest = RandomCutForest.builder().dimensions(tempDimensions).randomSeed(seed + 2)
                .outputAfter(50).shingleSize(shingleSize).sampleSize(sampleSize).internalShinglingEnabled(internal)
                .build();
        Random random = new Random(seed + 4);
        double score = 0;
        double error = 0;
        for (int i = 0; i < dataSize - 1; i++) {
            long timestamp = i * 100 + random.nextInt(20);
            if (mode != STREAMING_IMPUTE) {
                assertEquals(preprocessor.numberOfImputes(timestamp), 0);
            }
            PreprocessorMapper mapper = new PreprocessorMapper();
            Preprocessor newPre = mapper.toModel(mapper.toState(preprocessor));
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest);
            assertArrayEquals(shingle, newPre.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest));
            if (shingle != null && (mode != STREAMING_IMPUTE || i != 3 || random.nextDouble() > 0.1)) {
                if (i > 100 + shingleSize - 1) {
                    double currentScore = forest.getAnomalyScore(shingle);
                    if (currentScore > 1.5) {
                        float[] value = forest.imputeMissingValues(shingle, 1, new int[] { shingle.length - 1 });
                        double expected = preprocessor.getExpectedValue(0, dataWithKey.data[i], shingle, value)[0];
                        System.out.println(" expected " + expected + " in place of " + dataWithKey.data[i][0]);
                    }
                    score += currentScore;
                }
            }
            if (internal) {
                float[] input = preprocessor.getScaledInput(toFloatArray(dataWithKey.data[i]), timestamp);
                if (i != 20 && random.nextDouble() > 0.1) {
                    preprocessor.update(dataWithKey.data[i], input, timestamp, null, forest);
                } else {
                    // drop first coordinate
                    preprocessor.update(dataWithKey.data[i], input, timestamp, new int[1], forest);
                }
                if (shingleSize > 1) {
                    RangeVector rangeVector = forest.extrapolateFromShingle(preprocessor.getLastShingledPoint(), 1,
                            tempDimensions / shingleSize, 1.0);
                    TimedRangeVector timedRanges = preprocessor.invertForecastRange(rangeVector, timestamp, null, false,
                            timestamp);
                    // error of lookahead
                    if (i > 100 + shingleSize - 1) {
                        error += abs(timedRanges.rangeVector.values[0] - dataWithKey.data[i + 1][0]);
                    }
                }
            } else {
                // force two subsequent drops
                if (i != 0 && i != 5 && i != preprocessor.startNormalization - 1 && i != 500 && i != 501 && i != 502
                        && random.nextDouble() > 0.1) {
                    if (random.nextDouble() > 0.7) {
                        preprocessor.update(dataWithKey.data[i], shingle, timestamp, null, forest);
                    } else if (random.nextDouble() > 0.5) {
                        // same as null
                        preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[0], forest);
                    } else {
                        preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[] {}, forest);
                    }
                } else {
                    if (i != 5 && i != 6 && i != 500 && i != 501 && i != 502) {
                        // force initial; note 5 is dropped
                        preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[] { 0 }, forest);
                    }
                    // drop
                }
            }
        }
        assertTrue((score) / (dataSize - 100 - shingleSize + 1) < 1);
        // note for time-augmentation the noise in the time will overwhelm the noise in
        // the signal
        if (mode != TIME_AUGMENTED && (imputeMethod == null || imputeMethod == RCF)) {
            assertTrue((error) / (dataSize - 200 - shingleSize + 1) < 10); // twice the noise
        }
        PreprocessorMapper mapper = new PreprocessorMapper();
        Preprocessor second = mapper.toModel(mapper.toState(preprocessor));
        assertArrayEquals(second.getSmoothedDeviations(), preprocessor.getSmoothedDeviations(), 1e-10f);
        assertArrayEquals(second.getShift(), preprocessor.getShift(), 1e-10f);
        assertArrayEquals(second.getScale(), preprocessor.getScale(), 1e-10f);
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void preprocessorTest(TransformMethod method) {
        preprocessorPlusForest(0, STANDARD, method, null, true, 10);
        preprocessorPlusForest(0, STANDARD, method, null, true, 1);
        preprocessorPlusForest(0, STANDARD, method, null, false, 2);
        preprocessorPlusForest(0, STANDARD, method, RCF, false, 1);
        preprocessorPlusForest(0, TIME_AUGMENTED, method, null, true, 1);
        preprocessorPlusForest(0, TIME_AUGMENTED, method, null, true, 3);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, RCF, true, 10);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, PREVIOUS, false, 5);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ZERO, false, 11);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, FIXED_VALUES, true, 12);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, NEXT, false, 7);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, LINEAR, false, 2);
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void allMissing(TransformMethod method) {
        int dataSize = 1000;
        int shingleSize = 2;
        long seed = new Random().nextLong();
        Random random = new Random(seed + 4);
        double[] defaultFill = null;
        if (method == NORMALIZE) {
            defaultFill = new double[] { random.nextInt(10) };
        }

        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, 0, 1,
                false);
        Preprocessor.Builder<?> builder = Preprocessor.builder().inputLength(1).dimensions(shingleSize)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).transformMethod(method).randomSeed(seed + 1)
                .weightTime(0).normalizeTime(true).forestMode(STANDARD).fillValues(defaultFill);

        Preprocessor preprocessor = builder.build();

        // testing length of deviation list
        assertThrows(IllegalArgumentException.class, () -> preprocessor.manageDeviations(new Deviation[12], null, 0));

        for (int i = 0; i < preprocessor.getStartNormalization(); i++) {
            long timestamp = i * 100 + random.nextInt(20);
            PreprocessorMapper mapper = new PreprocessorMapper();
            Preprocessor newPre = mapper.toModel(mapper.toState(preprocessor));
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, null);
            assertArrayEquals(shingle, newPre.getScaledShingledInput(dataWithKey.data[i], timestamp, null, null));
            preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[] { 0 }, null);
        }
        assertTrue(preprocessor.getInitialValues() == null);
        assertTrue(preprocessor.isOutputReady());
        assertEquals(preprocessor.getScale().length, 1);
        assertEquals(preprocessor.getShift().length, 1);
        if (method == NORMALIZE) {
            assertTrue(preprocessor.getDeviationList()[0].getMean() == defaultFill[0]);
            assertEquals(preprocessor.getLastShingledInput()[0], defaultFill[0]);
        } else {
            assertTrue(preprocessor.getDeviationList()[0].getMean() == 0);
        }
        assertThrows(IllegalArgumentException.class, () -> preprocessor.inverseMapTime(0, -(shingleSize + 1)));
        assertEquals(preprocessor.inverseMapTimeValue(1L, 1L), 0);
        assertEquals(preprocessor.getShingledInput().length, shingleSize);
        assertEquals(preprocessor.dataQuality(), 0);
        assertTrue(preprocessor.normalize(-200, 1) < 0);
        assertEquals(preprocessor.getScale().length, 1);
        assertEquals(preprocessor.getShift().length, 1);
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void allMissingWithForest(TransformMethod method) {
        int dataSize = 1000;
        int shingleSize = 2;
        long seed = 0L;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, 0, 1,
                false);
        Preprocessor.Builder<?> builder = Preprocessor.builder().inputLength(1).dimensions(2 * shingleSize)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).transformMethod(method).randomSeed(seed + 1)
                .forestMode(TIME_AUGMENTED).weightTime(0).normalizeTime(false);
        RandomCutForest forest = new RandomCutForest.Builder().dimensions(2 * shingleSize)
                .internalShinglingEnabled(false) // not recommended
                .shingleSize(shingleSize).build();

        Preprocessor preprocessor = builder.build();
        Random random = new Random(seed + 4);
        assertTrue(!preprocessor.isOutputReady());
        assertThrows(IllegalArgumentException.class,
                () -> preprocessor.getScaledShingledInput(dataWithKey.data[0], 0, null, null));
        for (int i = 0; i < preprocessor.getStartNormalization() + 1; i++) {
            long timestamp = i * 100 + random.nextInt(20);
            PreprocessorMapper mapper = new PreprocessorMapper();
            Preprocessor newPre = mapper.toModel(mapper.toState(preprocessor));
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest);
            assertArrayEquals(shingle, newPre.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest));
            preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[] { 0 }, forest);
        }
        assertTrue(preprocessor.getInitialValues() == null);
        assertTrue(preprocessor.getDeviationList()[0].getMean() == 0);
        assertEquals(preprocessor.getScale().length, 2);
        assertEquals(preprocessor.getShift().length, 2);
        assertThrows(IllegalArgumentException.class, () -> preprocessor.inverseMapTime(0, -(shingleSize + 1)));
        assertEquals(preprocessor.inverseMapTimeValue(1L, 1L), 0);
        assertEquals(preprocessor.getShingledInput().length, shingleSize);
        assertEquals(preprocessor.dataQuality(), 0);
        assertTrue(preprocessor.normalize(0, 1) < 0);
        assertThrows(IllegalArgumentException.class, () -> preprocessor.invertForecastRange(new RangeVector(11),
                preprocessor.internalTimeStamp + 1, new double[0], false, 0));
        assertThrows(IllegalArgumentException.class, () -> preprocessor.invertForecastRange(new RangeVector(10),
                preprocessor.internalTimeStamp + 1, new double[0], false, 0));
        long[] values = preprocessor.invertForecastRange(new RangeVector(10), preprocessor.internalTimeStamp - 1,
                new double[1], false, -100).timeStamps;
        long[] otherValues = preprocessor.invertForecastRange(new RangeVector(10), preprocessor.internalTimeStamp - 1,
                new double[1], true, -100).timeStamps;
        assertTrue(values[0] == otherValues[0]);
        assertThrows(IllegalArgumentException.class, () -> preprocessor.invertInPlace(new float[10], null, -1));
        assertDoesNotThrow(() -> preprocessor.invertInPlace(new float[2], new double[1], -1));
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class, names = { "NONE", "WEIGHTED", "DIFFERENCE" })
    public void basicPreProcessor(TransformMethod method) {
        int dataSize = 1000;
        int shingleSize = 2;
        long seed = 0L;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, 0, 1,
                false);
        Preprocessor.Builder<?> builder = Preprocessor.builder().inputLength(1).dimensions(shingleSize)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).transformMethod(method).randomSeed(seed + 1)
                .forestMode(STANDARD).imputationMethod(ZERO);
        RandomCutForest forest = new RandomCutForest.Builder().dimensions(shingleSize).internalShinglingEnabled(false) // not
                // recommended
                .shingleSize(shingleSize).build();

        Preprocessor preprocessor = builder.build();
        Random random = new Random(seed + 4);

        assertDoesNotThrow(() -> preprocessor.getScaledShingledInput(dataWithKey.data[0], 0, null, null));
        for (int i = 0; i < preprocessor.getStartNormalization() + 1; i++) {
            long timestamp = i * 100 + random.nextInt(20);
            PreprocessorMapper mapper = new PreprocessorMapper();
            Preprocessor newPre = mapper.toModel(mapper.toState(preprocessor));
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest);
            assertArrayEquals(shingle, newPre.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest));
            preprocessor.update(dataWithKey.data[i], shingle, timestamp, new int[] { 0 }, forest);
        }
        assertTrue(preprocessor.getInitialValues() == null);
        assertTrue(preprocessor.getDeviationList()[0].getMean() == 0);
        assertThrows(IllegalArgumentException.class, () -> preprocessor.inverseMapTime(0, -(shingleSize + 1)));
        assertNotEquals(preprocessor.inverseMapTimeValue(1L, 1L), 0);
        assertEquals(preprocessor.getShingledInput().length, shingleSize);
        assertEquals(preprocessor.dataQuality(), 0);
        assertTrue(preprocessor.normalize(0, 1) < 0);
        assertThrows(IllegalArgumentException.class, () -> preprocessor.getExpectedValue(-2, null, null, new float[1]));
        assertThrows(IllegalArgumentException.class, () -> preprocessor.getExpectedValue(-2, null, null, new float[2]));
        preprocessor.getExpectedValue(-1, null, null, new float[2]);
        assertDoesNotThrow(() -> preprocessor.getExpectedValue(-1, null, null, new float[2]));
    }

    @ParameterizedTest
    @EnumSource(value = ImputationMethod.class)
    public void streamingImputeLargeGap(ImputationMethod method) {
        int dataSize = 1000;
        int shingleSize = 4;
        long seed = 0L;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, 0, 1,
                false);
        Preprocessor.Builder<?> builder = Preprocessor.builder().inputLength(1).dimensions(shingleSize)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).randomSeed(seed + 1)
                .forestMode(STREAMING_IMPUTE).imputationMethod(method).transformMethod(NORMALIZE).fastForward(true);
        if (method == FIXED_VALUES) {
            builder.fillValues(new double[] { 0 });
        }
        RandomCutForest forest = new RandomCutForest.Builder().dimensions(shingleSize).internalShinglingEnabled(true)
                .shingleSize(shingleSize).build();

        Preprocessor preprocessor = builder.build();
        Random random = new Random(seed + 4);

        assertDoesNotThrow(() -> preprocessor.getScaledShingledInput(dataWithKey.data[0], 0, null, null));
        for (int i = 0; i < dataSize; i++) {
            long timestamp = i * 100 + random.nextInt(20);
            PreprocessorMapper mapper = new PreprocessorMapper();
            Preprocessor newPre = mapper.toModel(mapper.toState(preprocessor));
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest);
            assertArrayEquals(shingle, newPre.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest));
            preprocessor.update(dataWithKey.data[i], shingle, timestamp, null, forest);
        }
        long updates = forest.getTotalUpdates();
        double[] newData = new double[] { -11.11 };
        float[] shingle = preprocessor.getScaledShingledInput(newData, 100 * dataSize + 10000L, null, forest);
        assertEquals(forest.getTotalUpdates(), updates);
        preprocessor.update(newData, shingle, 100 * dataSize + 10000L, null, forest);
        if (method == RCF) {
            assertEquals(forest.getTotalUpdates(), updates + shingleSize);
        } else {
            assertEquals(forest.getTotalUpdates(), updates + 100 + shingleSize / 2);
        }
    }

}
