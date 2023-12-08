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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.config.ForestMode.STANDARD;
import static com.amazon.randomcutforest.config.ForestMode.TIME_AUGMENTED;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.NEXT;
import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.config.TransformMethod.NONE;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.state.PredictiveRandomCutForestMapper;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData.NormalDistribution;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class PredictiveRandomCutForestTest {

    @Test
    public void testConfig() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        // have to enable internal shingling or keep it unspecified
        assertDoesNotThrow(
                () -> PredictiveRandomCutForest.builder().sampleSize(sampleSize).inputDimensions(baseDimensions)
                        .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).shingleSize(shingleSize).build());

        PredictiveRandomCutForest forest = PredictiveRandomCutForest.builder().sampleSize(sampleSize)
                .inputDimensions(baseDimensions).randomSeed(seed).startNormalization(1)
                .forestMode(ForestMode.TIME_AUGMENTED).shingleSize(shingleSize).build();
        assertNotNull(((Preprocessor) forest.getPreprocessor()).getInitialTimeStamps());
        assertEquals(forest.getForest().getDimensions(), (baseDimensions + 1) * shingleSize);
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).weights(new double[] { -1.0, 0.0 }).shingleSize(shingleSize).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).startNormalization(-10).shingleSize(shingleSize).threadPoolSize(1)
                        .build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).outputAfter(1).startNormalization(shingleSize + 10)
                        .shingleSize(shingleSize).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).shingleSize(shingleSize).transformMethod(NORMALIZE)
                        .startNormalization(111).stopNormalization(100).build());
    }

    public void simpleExample(int dataSize, TransformMethod method, ForestMode mode, double error) {
        int shingleSize = 1;
        int numberOfTrees = 100;
        int sampleSize = 256;

        // 5 dimensions, three are known and 4,5 th unknown (and stochastic)
        int baseDimensions = 5;

        PredictiveRandomCutForest forest = new PredictiveRandomCutForest.Builder<>().inputDimensions(baseDimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .forestMode(mode).startNormalization(32).transformMethod(method).build();

        long seed = 17;

        NormalDistribution normal = new NormalDistribution(new Random(seed));
        double total = 0;
        double extTotal = 0;
        Random random = new Random(seed + 10);
        for (int i = 0; i < dataSize; i++) {
            float[] record = generateRecordKey(random);
            checkArgument(record[3] == 0, " should not be filled");
            checkArgument(record[4] == 0, " should not be filled");

            SampleSummary answer = forest.predict(record, 0, new int[] { 3, 4 });
            assertEquals(answer.summaryPoints.length, answer.measure.length);
            fillInValues(record, random, normal);
            forest.update(record, 0);
            double tag = Double.MAX_VALUE;
            double ext = Double.MAX_VALUE;
            for (int y = 0; y < answer.summaryPoints.length; y++) {
                double t = Summarizer.L2distance(record, answer.summaryPoints[y]);
                double u = Summarizer.L2distance(new float[5], answer.measure[y]);
                if (t < tag) {
                    tag = t;
                    ext = u;
                }
            }

            if (i > forest.forest.getOutputAfter()) {
                total += tag;
                extTotal += ext;
            }
        }

        assertTrue(5 * error > total / (dataSize - forest.getForest().getOutputAfter()));
        assertTrue(5 * error > extTotal / (dataSize - forest.getForest().getOutputAfter()));

        PredictiveRandomCutForestMapper mapper = new PredictiveRandomCutForestMapper();
        PredictiveRandomCutForest second = mapper.toModel(mapper.toState(forest));
        assertArrayEquals(second.preprocessor.getLastShingledPoint(), forest.preprocessor.getLastShingledPoint(),
                1e-10f);
    }

    @Test
    public void configTest() {
        simpleExample(1000, NORMALIZE, STANDARD, 2);
        simpleExample(1000, NORMALIZE, TIME_AUGMENTED, 2);
        simpleExample(1000, NONE, STANDARD, 2);
        simpleExample(1000, NONE, TIME_AUGMENTED, 2);
    }

    float[] generateRecordKey(Random random) {
        float[] record = new float[5];
        double firstToss = random.nextDouble();
        double secondToss = random.nextDouble();
        double thirdToss = random.nextDouble();
        if (firstToss < 0.8) {
            record[0] = 1.0f;
            if (secondToss < 0.8) {
                record[1] = 19;
            } else {
                record[1] = 25;
            }
            record[2] = (float) thirdToss * 10;
        } else {
            record[0] = 0.0f;
            if (secondToss < 0.3) {
                record[1] = 16;
                record[2] = 12;
            } else {
                record[1] = 20;
                record[2] = 4;
            }
        }
        return record;
    }

    void fillInValues(float[] record, Random random, NormalDistribution normal) {
        if (record[0] < 0.5) {
            double next = random.nextDouble();
            record[3] = (float) ((next < 0.5) ? normal.nextDouble(20, 5) : normal.nextDouble(40, 5));
            record[4] = (float) normal.nextDouble(30, 3);
        } else {
            if (record[1] < 20) {
                record[3] = (float) normal.nextDouble(30, 10);
                record[4] = (float) normal.nextDouble(10, 3);
            } else {
                if (record[2] < 6) {
                    double next = random.nextDouble();
                    record[3] = (float) ((next < 0.3) ? normal.nextDouble(20, 5) : normal.nextDouble(40, 3));
                    record[4] = (float) normal.nextDouble(50, 1);
                } else {
                    double next = random.nextDouble();
                    record[3] = (float) normal.nextDouble(30, 1);
                    record[4] = (float) ((next < 0.7) ? normal.nextDouble(10, 3) : normal.nextDouble(30, 5));
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(ImputationMethod.class)
    void testImpute(ImputationMethod method) {
        int baseDimensions = 1;

        // long seed = new Random().nextLong();

        // shingle size 1 ie not useful for impute
        assertThrows(IllegalArgumentException.class, () -> {
            PredictiveRandomCutForest forest = PredictiveRandomCutForest.builder().inputDimensions(baseDimensions)
                    .randomSeed(0).forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(method).shingleSize(1)
                    .build();
        });

        int newShingleSize = 4;

        PredictiveRandomCutForest forest = PredictiveRandomCutForest.builder().inputDimensions(baseDimensions)
                .randomSeed(42).forestMode(ForestMode.STREAMING_IMPUTE).imputationMethod(method)
                .transformMethod(NORMALIZE).storeSequenceIndexesEnabled(true).shingleSize(newShingleSize)
                .useImputedFraction(0.76).fillValues(new double[] { 0 }).build();

        float[] fixedData = new float[] { 1.0f };
        float[] newData = new float[] { 10.0f };
        float[] negativeData = new float[] { -10.0f };
        Random random = new Random(0);
        int count = 0;
        for (int i = 0; i < 200 + new Random().nextInt(100); i++) {
            long timeStamp = (long) count * 113 + random.nextInt(10);
            float[] test = (random.nextDouble() < 0.5) ? newData : negativeData;
            double scoreA = forest.getExpectedInverseDepthScore(test, timeStamp);
            assertTrue(scoreA == 0.0 || scoreA > 2.0);
            double scoreB = forest.getExpectedInverseDepthAttribution(test, timeStamp).getHighLowSum();
            assertEquals(scoreA, scoreB, 1e-6);
            double scoreC = forest.getRCFDistanceAttribution(test, timeStamp).getHighLowSum();
            assertTrue(scoreC == 0.0 || scoreC > 8.0);
            if (i != 20 && random.nextDouble() < 0.9) {
                // few drops -- and definitely one during normalization
                forest.update(fixedData, timeStamp);
            } else {
                // note that the large should be imputed away
                forest.update(test, timeStamp, new int[] { 0 });
            }
            ++count;
        }

        long timestamp = (long) count * 113 + 1000;
        double score = forest.getExpectedInverseDepthScore(newData, timestamp);
        assertEquals(score, forest.getExpectedInverseDepthAttribution(newData, timestamp).getHighLowSum(), 1e-6);
        assertTrue(score > 1.0);
        if (method != NEXT && method != ZERO && method != FIXED_VALUES) {
            if (method == RCF) {
                SampleSummary summary = forest.predict(newData, timestamp, new int[] { 0 });
                assertArrayEquals(summary.summaryPoints[0], fixedData, 1e-6f);
            }
        }
        assertEquals(forest.getForest().getTotalUpdates(), count);
        // the next gap is 1226 + 113 which is about 11 times 113
        long newstamp = (long) count * 113 + 1226;
        assertEquals(11, forest.preprocessor.numberOfImputes(newstamp));
        forest.update(newData, newstamp);

        // time has to increase for streamingImpute
        assertThrows(IllegalArgumentException.class, () -> {
            forest.update(newData, newstamp - 1);
        });
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void timeAugmentedTest(TransformMethod transformMethod) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;

        int numTrials = 1; // test is exact equality, reducing the number of trials
        int numberOfTrees = 30; // and using fewer trees to speed up test
        int length = 10 * sampleSize;
        int dataSize = 2 * length;
        for (int i = 0; i < numTrials; i++) {
            long seed = new Random().nextLong();
            System.out.println("seed = " + seed);

            PredictiveRandomCutForest first = PredictiveRandomCutForest.builder().inputDimensions(baseDimensions)
                    .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                    .forestMode(ForestMode.STANDARD).transformMethod(transformMethod).outputAfter(32)
                    .initialAcceptFraction(0.125).build();
            PredictiveRandomCutForest second = PredictiveRandomCutForest.builder().inputDimensions(baseDimensions)
                    .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                    .forestMode(ForestMode.TIME_AUGMENTED).weightTime(0).transformMethod(transformMethod)
                    .outputAfter(32).initialAcceptFraction(0.125).build();

            Random noise = new Random(0);

            // change the last argument seed for a different run
            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1,
                    50, 100, 5, seed, baseDimensions);

            int count = 0;
            for (int j = 0; j < length; j++) {

                long timestamp = 100 * count + noise.nextInt(10) - 5;
                assertEquals(first.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp),
                        second.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp));
                first.update(toFloatArray(dataWithKeys.data[j]), timestamp);
                second.update(toFloatArray(dataWithKeys.data[j]), timestamp);
                // grade will not be the same because dimension changes
                ++count;
            }

            PredictiveRandomCutForestMapper mapper = new PredictiveRandomCutForestMapper();
            PredictiveRandomCutForest third = mapper.toModel(mapper.toState(second));
            for (int j = length; j < 2 * length; j++) {

                // can be a different gap
                long timestamp = 150 * count + noise.nextInt(10) - 5;
                assertEquals(first.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp),
                        second.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp));
                assertEquals(first.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp),
                        third.getExpectedInverseDepthScore(toFloatArray(dataWithKeys.data[j]), timestamp));
                first.update(toFloatArray(dataWithKeys.data[j]), timestamp);
                second.update(toFloatArray(dataWithKeys.data[j]), timestamp);
                third.update(toFloatArray(dataWithKeys.data[j]), timestamp);
            }
        }
    }
}
