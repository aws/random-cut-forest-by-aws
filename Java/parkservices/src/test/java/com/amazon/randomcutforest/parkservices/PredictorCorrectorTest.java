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

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static com.amazon.randomcutforest.parkservices.PredictorCorrector.DEFAULT_SAMPLING_SUPPORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorMapper;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.statistics.Deviation;

public class PredictorCorrectorTest {

    @Test
    void AttributorTest() {
        int sampleSize = 256;
        int baseDimensions = 10;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01).transformMethod(NORMALIZE)
                .build();
        DiVector test = new DiVector(baseDimensions * shingleSize);
        assert (forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null) == null);
        assertThrows(IllegalArgumentException.class, () -> forest.predictorCorrector.setNumberOfAttributors(-1));
        forest.predictorCorrector.setNumberOfAttributors(baseDimensions);
        assertThrows(NullPointerException.class,
                () -> forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null));
        double[] array = new double[20];
        Arrays.fill(array, 1.0);
        DiVector testTwo = new DiVector(array, array);
        assertThrows(NullPointerException.class,
                () -> forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null));
    }

    @Test
    void configTest() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        double[] testOne = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testTwo = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testThree = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testFour = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                .scoringStrategy(ScoringStrategy.DISTANCE).transformMethod(NORMALIZE).randomSeed(1110).autoAdjust(true)
                .ignoreNearExpectedFromAbove(testOne).ignoreNearExpectedFromBelow(testTwo)
                .ignoreNearExpectedFromAboveByRatio(testThree).ignoreNearExpectedFromBelowByRatio(testFour).build();
        PredictorCorrector predictorCorrector = forest.getPredictorCorrector();
        assertEquals(predictorCorrector.getSamplingSupport(), DEFAULT_SAMPLING_SUPPORT);
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setSamplingSupport(-1.0));
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setSamplingSupport(2.0));
        assertDoesNotThrow(() -> predictorCorrector.setSamplingSupport(1.5 * DEFAULT_SAMPLING_SUPPORT));
        double[] test = new double[1];
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setIgnoreNearExpected(test));
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpected(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromAbove(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromBelow(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromAboveByRatio(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromBelowByRatio(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(predictorCorrector.getDeviations());
        assertEquals(predictorCorrector.lastStrategy, ScoringStrategy.DISTANCE);
        assertThrows(IllegalArgumentException.class,
                () -> predictorCorrector.getCachedAttribution(1, null, new DiVector[2], null));

        PredictorCorrectorMapper mapper = new PredictorCorrectorMapper();
        PredictorCorrector copy = mapper.toModel(mapper.toState(predictorCorrector));
        assertArrayEquals(copy.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(copy.getDeviations());
        assertEquals(copy.lastStrategy, ScoringStrategy.DISTANCE);
        copy.deviationsActual = new Deviation[1]; // changing the state
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> copy.getDeviations());
        assertEquals("incorrect state", exception.getMessage());
        copy.deviationsExpected = new Deviation[1];
        exception = assertThrows(IllegalArgumentException.class, () -> copy.getDeviations());
        assertEquals("length should be base dimension", exception.getMessage());

        double[] another = new double[4 * baseDimensions];
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpected(another));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, new double[2]);
        another[0] = -1;
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setIgnoreNearExpected(another));
        forest.setIgnoreNearExpectedFromAbove(testOne);
        forest.setIgnoreNearExpectedFromBelow(testTwo);
        forest.setIgnoreNearExpectedFromAboveByRatio(testThree);
        forest.setIgnoreNearExpectedFromBelowByRatio(testFour);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);

        Random testRandom = new Random(1110L);
        assertEquals(predictorCorrector.getRandomSeed(), 1110L);
        double nextDouble = predictorCorrector.nextDouble();
        assertEquals(predictorCorrector.getRandomSeed(), testRandom.nextLong());
        assertEquals(nextDouble, testRandom.nextDouble(), 1e-10);

    }

    @Test
    public void mapperTest() {
        assertThrows(IllegalArgumentException.class, () -> new PredictorCorrector(new BasicThresholder[0], null, 1, 0));
        assertThrows(NullPointerException.class, () -> new PredictorCorrector(new BasicThresholder[1], null, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new PredictorCorrector(new BasicThresholder[] { new BasicThresholder(0) }, new Deviation[1], 1,
                        0));
    }

    @Test
    public void expectedValueTest() {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(20).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(1).anomalyRate(0.01)
                .scoringStrategy(ScoringStrategy.DISTANCE).transformMethod(NORMALIZE).build();
        PredictorCorrector predictorCorrector = forest.getPredictorCorrector();
        double[] vector = new double[20];
        Arrays.fill(vector, 1.0);
        DiVector diVec = new DiVector(vector, vector);
        assertNull(predictorCorrector.getExpectedPoint(diVec, 0, 20, null, null));
        assertTrue(predictorCorrector.trigger(diVec, 1, 20, null, null, 1.0));
        assertTrue(predictorCorrector.trigger(diVec, 21, 20, null, null, 1.0));
        assertTrue(predictorCorrector.trigger(diVec, 21, 20, diVec, null, 1.0));
        assertEquals(1, predictorCorrector.centeredTransformPass(new AnomalyDescriptor(null, 0), toFloatArray(vector)));
        Arrays.fill(vector, 0);
        assertEquals(0, predictorCorrector.centeredTransformPass(new AnomalyDescriptor(null, 0), toFloatArray(vector)));
    }

    @Test
    public void runLengthTest() {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(4).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(4).anomalyRate(0.01).autoAdjust(false)
                .scoringStrategy(ScoringStrategy.MULTI_MODE).transformMethod(NORMALIZE).build();
        for (int i = 0; i < 100; i++) {
            forest.process(new double[] { 10 }, 0);
        }
        for (int i = 0; i < 100; i++) {
            forest.process(new double[] { 20 }, 0);
        }
        double[] scores = forest.getPredictorCorrector().getLastScore();
        forest.predictorCorrector.setLastScore(null);
        assertArrayEquals(forest.predictorCorrector.getLastScore(), scores, 1e-10);
    }
}
