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

import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorMapper;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.DiVector;

public class PredictorCorrectorTest {

    @Test
    void AttributorTest() {
        int sampleSize = 256;
        int baseDimensions = 10;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(0L).forestMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).transformMethod(NORMALIZE).build();
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
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(0L).forestMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).scoringStrategy(ScoringStrategy.DISTANCE).transformMethod(NORMALIZE).randomSeed(1110)
                .learnIgnoreNearExpected(true).ignoreNearExpectedFromAbove(testOne).ignoreNearExpectedFromBelow(testTwo)
                .ignoreNearExpectedFromAboveByRatio(testThree).ignoreNearExpectedFromBelowByRatio(testFour).build();
        PredictorCorrector predictorCorrector = forest.getPredictorCorrector();
        double[] test = new double[1];
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setIgnoreNearExpected(test));
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpected(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(predictorCorrector.getDeviations());
        assertEquals(predictorCorrector.lastStrategy, ScoringStrategy.DISTANCE);

        PredictorCorrectorMapper mapper = new PredictorCorrectorMapper();
        PredictorCorrector copy = mapper.toModel(mapper.toState(predictorCorrector));
        assertArrayEquals(copy.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(copy.getDeviations());
        assertEquals(copy.lastStrategy, ScoringStrategy.DISTANCE);
        copy.deviationsAbove = new Deviation[1]; // changing the state
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> copy.getDeviations());
        assertEquals("incorrect state", exception.getMessage());
        copy.deviationsBelow = new Deviation[1];
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
        predictorCorrector.setIgnoreNearExpectedFromAbove(testOne);
        predictorCorrector.setIgnoreNearExpectedFromBelow(testTwo);
        predictorCorrector.setIgnoreNearExpectedFromAboveByRatio(testThree);
        predictorCorrector.setIgnoreNearExpectedFromBelowByRatio(testFour);
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

}
