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
import static com.amazon.randomcutforest.config.ForestMode.STANDARD;
import static com.amazon.randomcutforest.config.ForestMode.TIME_AUGMENTED;
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

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.state.PredictiveRandomCutForestMapper;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData.NormalDistribution;

public class PredictiveRandomCutForestTest {

    @Test
    public void testConfig() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 4;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().sampleSize(sampleSize).inputDimensions(baseDimensions)
                        .randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(false)
                        .shingleSize(shingleSize).anomalyRate(0.01).build());

        // have to enable internal shingling or keep it unspecified
        assertDoesNotThrow(() -> PredictiveRandomCutForest.builder().sampleSize(sampleSize)
                .inputDimensions(baseDimensions).randomSeed(seed).forestMode(ForestMode.TIME_AUGMENTED)
                .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build());

        PredictiveRandomCutForest forest = PredictiveRandomCutForest.builder().sampleSize(sampleSize)
                .inputDimensions(baseDimensions).randomSeed(seed).startNormalization(1)
                .forestMode(ForestMode.TIME_AUGMENTED).shingleSize(shingleSize).anomalyRate(0.01).build();
        assertNotNull(((Preprocessor) forest.getPreprocessor()).getInitialTimeStamps());
        assertEquals(forest.getForest().getDimensions(), (baseDimensions + 1) * shingleSize);
        assertThrows(IllegalArgumentException.class,
                () -> new PredictiveRandomCutForest.Builder<>().inputDimensions(dimensions).randomSeed(seed)
                        .forestMode(ForestMode.TIME_AUGMENTED).internalShinglingEnabled(false).shingleSize(shingleSize)
                        .transformMethod(NORMALIZE).startNormalization(1).anomalyRate(0.01).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).internalShinglingEnabled(false).weights(new double[] { -1.0, 0.0 })
                        .shingleSize(shingleSize).anomalyRate(0.01).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).internalShinglingEnabled(false).startNormalization(-10)
                        .shingleSize(shingleSize).anomalyRate(0.01).threadPoolSize(1).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).internalShinglingEnabled(false).outputAfter(1)
                        .startNormalization(shingleSize + 10).shingleSize(shingleSize).anomalyRate(0.01).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(baseDimensions).randomSeed(seed)
                        .forestMode(STANDARD).shingleSize(shingleSize).anomalyRate(0.01).transformMethod(NORMALIZE)
                        .startNormalization(111).stopNormalization(100).build());
        assertThrows(IllegalArgumentException.class,
                () -> PredictiveRandomCutForest.builder().inputDimensions(dimensions).randomSeed(seed)
                        .forestMode(ForestMode.STREAMING_IMPUTE).internalShinglingEnabled(true).shingleSize(shingleSize)
                        .anomalyRate(0.01).build());

    }

    public void simpleExample(boolean internal, int dataSize, TransformMethod method, ForestMode mode, double error) {
        int shingleSize = 1;
        int numberOfTrees = 100;
        int sampleSize = 256;

        // 5 dimensions, three are known and 4,5 th unknown (and stochastic)
        int baseDimensions = 5;

        PredictiveRandomCutForest forest = new PredictiveRandomCutForest.Builder<>().inputDimensions(baseDimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(internal).startNormalization(3).transformMethod(method).build();

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
        simpleExample(true, 1000, NORMALIZE, STANDARD, 1);
        simpleExample(false, 1000, NORMALIZE, STANDARD, 1);
        simpleExample(true, 1000, NORMALIZE, TIME_AUGMENTED, 1);
        simpleExample(false, 1000, NONE, STANDARD, 1);
        simpleExample(true, 1000, NONE, STANDARD, 1);
        simpleExample(true, 1000, NONE, TIME_AUGMENTED, 1);
        simpleExample(false, 1000, NONE, TIME_AUGMENTED, 1);
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
}
