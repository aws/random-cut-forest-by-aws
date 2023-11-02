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
import static com.amazon.randomcutforest.config.ForestMode.STREAMING_IMPUTE;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.TransformMethod.NONE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE_DIFFERENCE;
import static java.lang.Math.abs;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class PreprocessorTest {

    @Test
    void builderTest() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(null).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(null).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.STANDARD).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.STANDARD).inputLength(10).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.STANDARD).inputLength(10)
                    .dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.STANDARD).inputLength(12)
                    .dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.TIME_AUGMENTED).inputLength(12)
                    .dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.TIME_AUGMENTED).inputLength(12)
                    .dimensions(13).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(12).dimensions(13).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(12).dimensions(14).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(6).dimensions(14).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(-2).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(6).dimensions(14).build();
        });

        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).normalizeTime(true)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(6).dimensions(14).build();
        });

        // external shingling in STANDARD mode
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(6).dimensions(14).build();
        });

        // internal shingling
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.STANDARD)
                    .inputLength(6).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.STANDARD)
                    .weights(new double[1]).inputLength(6).dimensions(12).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.STANDARD)
                    .weights(new double[2]).inputLength(6).dimensions(12).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.STANDARD)
                    .weights(new double[] { 1.0, 1.0 }).inputLength(6).dimensions(12).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).forestMode(ForestMode.STANDARD).inputLength(6)
                    .dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).normalizeTime(true)
                    .forestMode(ForestMode.STANDARD).inputLength(6).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).shingleSize(2).forestMode(ForestMode.STANDARD)
                    .inputLength(5).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).inputLength(5).dimensions(5).startNormalization(0)
                    .normalizeTime(true).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(NONE).inputLength(1).dimensions(1)
                    .weights(new double[] { 0.5 }).build();
        });
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(1)
                .startNormalization(0).transformMethod(NORMALIZE_DIFFERENCE).build());
        assertThrows(IllegalArgumentException.class, () -> new Preprocessor.Builder<>().inputLength(1).dimensions(1)
                .startNormalization(0).transformMethod(NORMALIZE_DIFFERENCE).build());
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().inputLength(1).dimensions(1).forestMode(STREAMING_IMPUTE)
                    .imputationMethod(FIXED_VALUES).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().inputLength(1).dimensions(1).forestMode(STREAMING_IMPUTE)
                    .imputationMethod(FIXED_VALUES).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().inputLength(1).dimensions(1).forestMode(STREAMING_IMPUTE)
                    .imputationMethod(FIXED_VALUES).fillValues(new double[2]).build();
        });
    }

    public void preprocessorPlusForest(int seed, ForestMode mode, TransformMethod method, ImputationMethod imputeMethod,
            boolean internalShinglingHint) {
        int dataSize = 1000;
        int shingleSize = 20;
        int sampleSize = 256;
        int tempDimensions = (mode == ForestMode.TIME_AUGMENTED) ? 2 * shingleSize : shingleSize;
        MultiDimDataWithKey dataWithKey = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 50, 5, seed, 1,
                false);
        Preprocessor.Builder builder = Preprocessor.builder().inputLength(1).dimensions(tempDimensions)
                .weights(new double[] { 1.0 }).shingleSize(shingleSize).transformMethod(method).randomSeed(seed + 1)
                .forestMode(mode);
        if (mode == STREAMING_IMPUTE) {
            builder.imputationMethod(imputeMethod);
            if (imputeMethod == FIXED_VALUES) {
                builder.fillValues(new double[] { 5 });
            }
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
            float[] shingle = preprocessor.getScaledShingledInput(dataWithKey.data[i], timestamp, null, forest);
            if (shingle != null && (mode != STREAMING_IMPUTE || i != 3 || random.nextDouble() > 0.1)) {
                if (i > 50 + shingleSize - 1) {
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
                preprocessor.update(dataWithKey.data[i], input, timestamp, null, forest);
                RangeVector rangeVector = forest.extrapolateFromShingle(preprocessor.getLastShingledPoint(), 1,
                        tempDimensions / shingleSize, 1.0);
                TimedRangeVector timedRanges = preprocessor.invertForecastRange(rangeVector, timestamp, null, false,
                        timestamp);
                // error of lookahead
                if (i > 100 + shingleSize - 1) {
                    error += abs(timedRanges.rangeVector.values[0] - dataWithKey.data[i + 1][0]);
                }
            } else {
                // following will update the forest -- for impute it would update multiple times
                preprocessor.update(dataWithKey.data[i], shingle, timestamp, null, forest);
            }

        }
        assertTrue((score) / (dataSize - 50 - shingleSize + 1) < 1);
        // note for time-augmentation the noise in the time will overwhelm the noise in
        // the signal
        if (mode != ForestMode.TIME_AUGMENTED && (imputeMethod == null || imputeMethod == ImputationMethod.RCF)) {
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
        preprocessorPlusForest(0, ForestMode.STANDARD, method, null, true);
        preprocessorPlusForest(0, ForestMode.STANDARD, method, null, false);
        preprocessorPlusForest(0, ForestMode.TIME_AUGMENTED, method, null, true);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ImputationMethod.RCF, false);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ImputationMethod.PREVIOUS, false);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ImputationMethod.ZERO, false);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, FIXED_VALUES, false);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ImputationMethod.NEXT, false);
        preprocessorPlusForest(0, STREAMING_IMPUTE, method, ImputationMethod.LINEAR, false);
    }
}
