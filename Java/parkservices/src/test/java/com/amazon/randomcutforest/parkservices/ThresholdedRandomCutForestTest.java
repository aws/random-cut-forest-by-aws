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

import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.NEXT;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.config.TransformMethod.DIFFERENCE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE_DIFFERENCE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;

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
                        .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());

        // have to enable internal shingling or keep it unspecified
        assertDoesNotThrow(
                () -> ThresholdedRandomCutForest.builder().compact(true).sampleSize(sampleSize).dimensions(dimensions)
                        .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED)
                        .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build());

        // imputefraction not allowed
        assertThrows(IllegalArgumentException.class,
                () -> new ThresholdedRandomCutForest.Builder<>().compact(true).sampleSize(sampleSize)
                        .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                        .forestMode(ForestMode.TIME_AUGMENTED).useImputedFraction(0.5).internalShinglingEnabled(true)
                        .shingleSize(shingleSize).anomalyRate(0.01).build());

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).sampleSize(sampleSize)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(true).shingleSize(shingleSize).anomalyRate(0.01)
                .build();
        assertNotNull(((Preprocessor) forest.getPreprocessor()).getInitialTimeStamps());
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
                    .forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(false).shingleSize(shingleSize)
                    .anomalyRate(0.01).build();
            assertEquals(forest.getForest().getDimensions(), dimensions + 1);

        });

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                .forestMode(ForestMode.TIME_AUGMENTED).shingleSize(shingleSize).anomalyRate(0.01).build();
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
                        .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE)
                        .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build());

        assertDoesNotThrow(() -> new ThresholdedRandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE)
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
                        .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STANDARD)
                        .useImputedFraction(0.5).internalShinglingEnabled(false).shingleSize(shingleSize)
                        .anomalyRate(0.01).build());

        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions).precision(Precision.FLOAT_32)
                    .randomSeed(seed).forestMode(ForestMode.STANDARD).internalShinglingEnabled(false)
                    .shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                    .transformMethod(NORMALIZE).startNormalization(111).stopNormalization(100).build();
        });
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).transformMethod(NORMALIZE).startNormalization(111).stopNormalization(111).build();

        assertFalse(forest.getForest().isInternalShinglingEnabled()); // left to false
        assertEquals(((Preprocessor) forest.getPreprocessor()).getInitialValues().length, 111);
        assertEquals(((Preprocessor) forest.getPreprocessor()).getInitialTimeStamps().length, 111);
        assertEquals(((Preprocessor) forest.getPreprocessor()).getStopNormalization(), 111);
        assertEquals(((Preprocessor) forest.getPreprocessor()).getStartNormalization(), 111);
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
                    .forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(ImputationMethod.FIXED_VALUES)
                    .normalizeTime(true).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                    .build();
        });

        // incorrect number of values to fill
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(ImputationMethod.FIXED_VALUES)
                    .fillValues(new double[] { 0.0, 17.0 }).normalizeTime(true).internalShinglingEnabled(true)
                    .shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(ImputationMethod.FIXED_VALUES)
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

        // shingle size 1 ie not useful for impute
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(method).normalizeTime(true)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();
        });

        int newShingleSize = 4;
        int newDimensions = baseDimensions * newShingleSize;

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(newDimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE)
                .imputationMethod(method).internalShinglingEnabled(true).shingleSize(newShingleSize).anomalyRate(0.01)
                .useImputedFraction(0.76).fillValues(new double[] { 0 }).build();

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random(0);
        int count = 0;
        for (int i = 0; i < 200 + new Random().nextInt(100); i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        assert (result.getAnomalyGrade() > 0);
        assert (result.isExpectedValuesPresent());
        if (method != NEXT && method != ZERO && method != FIXED_VALUES) {
            assert (result.getRelativeIndex() == 0);
            assertArrayEquals(result.getExpectedValuesList()[0], fixedData, 1e-6);
        }
        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one arise from the actual input
        assertEquals(forest.getForest().getTotalUpdates(), count + 1);
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

    @ParameterizedTest
    @MethodSource("args")
    void testImpute(TransformMethod transformMethod, ImputationMethod method) {
        int baseDimensions = 1;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STREAMING_IMPUTE)
                .imputationMethod(method).internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01)
                .useImputedFraction(0.76).fillValues(new double[] { 1.0 }).transformMethod(transformMethod).build();

        double[] fixedData = new double[] { 1.0 };
        double[] newData = new double[] { 10.0 };
        Random random = new Random();
        int count = 0;
        for (int i = 0; i < 2000 + new Random().nextInt(100); i++) {
            forest.process(fixedData, (long) count * 113 + random.nextInt(10));
            ++count;
        }

        // note every will have an update
        assertEquals(forest.getForest().getTotalUpdates() + shingleSize - 1, count);
        AnomalyDescriptor result = forest.process(newData, (long) count * 113 + 1000);
        if (method != NEXT) {
            assert (result.getAnomalyGrade() > 0);
            assert (result.isExpectedValuesPresent());
        }
        // the other impute methods generate too much noise
        if (method == RCF || method == PREVIOUS) {
            assert (Math.abs(result.getExpectedValuesList()[0][0] - fixedData[0]) < 0.05);
        }

        // the gap is 1000 + 113 which is about 9 times 113
        // but only the first three entries are allowed in with shinglesize 4,
        // after which the imputation is 100% and
        // only at most 76% imputed tuples are allowed in the forest
        // an additional one does not arise from the actual input because all the
        // initial
        // entries are imputed and the method involves differencing
        if (transformMethod != DIFFERENCE && transformMethod != NORMALIZE_DIFFERENCE) {
            assertEquals(forest.getForest().getTotalUpdates(), count + 1);
        } else {
            assertEquals(forest.getForest().getTotalUpdates(), count);
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
