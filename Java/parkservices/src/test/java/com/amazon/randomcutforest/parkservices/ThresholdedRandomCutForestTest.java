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
import static com.amazon.randomcutforest.config.ImputationMethod.LINEAR;
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
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;

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
        // change if baseDimension != 2
        double[] testOne = new double[] { 0 };
        double[] testTwo = new double[] { 0, -1 };
        double[] testThree = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testFour = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testFive = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testSix = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                    .transformMethod(NORMALIZE).ignoreNearExpectedFromAbove(testOne).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                    .transformMethod(NORMALIZE).ignoreNearExpectedFromAbove(testTwo).build();
        });
        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                    .transformMethod(NORMALIZE).ignoreNearExpectedFromAbove(testThree)
                    .ignoreNearExpectedFromBelow(testFour).ignoreNearExpectedFromAboveByRatio(testFive)
                    .ignoreNearExpectedFromBelowByRatio(testSix).build();
            double[] array = forest.getPredictorCorrector().getIgnoreNearExpected();
            assert (array.length == 4 * baseDimensions);
            assert (array[0] == testThree[0]);
            assert (array[1] == testThree[1]);
            assert (array[2] == testFour[0]);
            assert (array[3] == testFour[1]);
            assert (array[4] == testFive[0]);
            assert (array[5] == testFive[1]);
            assert (array[6] == testSix[0]);
            assert (array[7] == testSix[1]);
            double random = new Random().nextDouble();
            assertThrows(IllegalArgumentException.class, () -> forest.predictorCorrector.setSamplingRate(-1));
            assertDoesNotThrow(() -> forest.predictorCorrector.setSamplingRate(random));
            assertEquals(forest.predictorCorrector.getSamplingRate(), random, 1e-10);
            long newSeed = forest.predictorCorrector.getRandomSeed();
            assertEquals(new Random(seed).nextLong(), newSeed);
            assertFalse(forest.predictorCorrector.autoAdjust);
            assertNull(forest.predictorCorrector.getDeviations());
        });
        assertDoesNotThrow(() -> {
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                    .transformMethod(NORMALIZE).learnIgnoreNearExpected(true).build();
            assertTrue(forest.predictorCorrector.autoAdjust);
            assert (forest.predictorCorrector.getDeviations().length == 2 * baseDimensions);
        });
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).forestMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).transformMethod(NORMALIZE).startNormalization(111).stopNormalization(111).build();

        assertTrue(forest.getForest().isInternalShinglingEnabled()); // left to false
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
        // triggerring consecutive anomalies (no differencing)
        // Note NEXT and LINEAR will have an obvious issue with consecutive anomalies
        if (method != NEXT && method != LINEAR) {
            assertEquals(forest.process(newData, (long) count * 113 + 1113).getAnomalyGrade(), 1.0);
        }
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

    @Test
    void testMapper() {
        double[] initialData = new double[] { 25.0, 25.0, 25.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 23.0, 23.0,
                23.0, 23.0, 23.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 21.0, 20.0, 20.0,
                20.0, 20.0, 20.0, 20.0, 20.0, 19.0, 19.0, 19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                17.0, 17.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 15.0, 15.0, 15.0, 15.0, 15.0,
                15.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0,
                18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0,
                23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0,
                23.0, 23.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0,
                21.0, 21.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
                19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0,
                17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                16.0, 16.0, 16.0, 16.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
                15.0, 15.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0,
                16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0,
                20.0, 20.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0,
                25.0, 25.0, 25.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0, 28.0, 28.0, 29.0, 29.0,
                29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0,
                28.0, 28.0, 28.0, 28.0, 28.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0,
                27.0, 27.0, 27.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
                25.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0,
                27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0,
                28.0, 28.0, 28.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 25.0,
                25.0, 25.0, 25.0, 25.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 23.0, 23.0, 23.0, 23.0, 23.0, 22.0,
                22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 21.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                20.0, 19.0, 19.0, 19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
                19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
                19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 21.0, 21.0,
                21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0,
                22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                22.0, 22.0, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0,
                21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
                19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 17.0, 17.0,
                17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 15.0,
                15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 13.0,
                13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 13.0, 13.0, 13.0,
                14.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 18.0,
                18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 21.0, 21.0, 21.0, 22.0, 22.0,
                22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0, 25.0, 25.0, 25.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0,
                27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 25.0, 25.0, 25.0, 25.0, 24.0, 24.0, 24.0, 24.0, 24.0, 23.0,
                23.0, 23.0, 23.0, 23.0, 22.0, 22.0, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 21.0, 20.0, 20.0, 20.0, 20.0,
                20.0, 19.0, 19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0, 17.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0,
                16.0, 16.0, 15.0, 15.0, 15.0, 15.0, 15.0, 14.0, 14.0, 14.0, 14.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,
                13.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
                15.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0,
                17.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
                19.0, 20.0, 20.0, 20.0, 20.0, 20.0 };

        double[] data = new double[] { 13.0, 20.0, 26.0, 18.0 };

        int shingleSize = 8;
        int numberOfTrees = 30;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;

        int baseDimensions = 1;
        long seed = -3095522926185205814L;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(seed).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .precision(precision).parallelExecutionEnabled(false).outputAfter(32).internalShinglingEnabled(true)
                .anomalyRate(0.005).initialAcceptFraction(0.125).timeDecay(0.0001).boundingBoxCacheFraction(0)
                .forestMode(ForestMode.STANDARD).build();

        double scoreSum = 0;

        for (double dataPoint : initialData) {
            AnomalyDescriptor result = forest.process(new double[] { dataPoint }, 0L);
            scoreSum += result.getRCFScore();
        }

        // checking average score < 1
        assert (scoreSum < initialData.length);

        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ThresholdedRandomCutForest second = mapper.toModel(mapper.toState(forest));

        for (double dataPoint : data) {
            AnomalyDescriptor result = second.process(new double[] { dataPoint }, 0L);
            // average score jumps due to discontinuity, checking > 1
            assert (result.getRCFScore() > 1.0);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6 })
    void smallGap(int gap) {
        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;
        // 10 trials each
        int numTrials = 10;

        int correct = 0;
        for (int z = 0; z < numTrials; z++) {
            int dimensions = baseDimensions * shingleSize;
            TransformMethod transformMethod = TransformMethod.NORMALIZE;
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true)
                    .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                    .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                    .transformMethod(transformMethod).build();

            long seed = new Random().nextLong();
            System.out.println("seed = " + seed);
            Random rng = new Random(seed);
            for (int i = 0; i < dataSize; i++) {
                double[] point = new double[] { 0.6 + 0.2 * (2 * rng.nextDouble() - 1) };
                AnomalyDescriptor result = forest.process(point, 0L);
            }
            AnomalyDescriptor result = forest.process(new double[] { 11.2 }, 0L);
            for (int y = 0; y < gap; y++) {
                result = forest.process(new double[] { 0.6 + 0.2 * (2 * rng.nextDouble() - 1) }, 0L);
            }
            result = forest.process(new double[] { 10.0 }, 0L);
            if (result.getAnomalyGrade() > 0) {
                ++correct;
            }
            ;
        }
        assert (correct > 0.9 * numTrials);
    }
}
