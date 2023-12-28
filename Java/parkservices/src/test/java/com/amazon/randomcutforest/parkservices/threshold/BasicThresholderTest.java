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

package com.amazon.randomcutforest.parkservices.threshold;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.statistics.Deviation;

public class BasicThresholderTest {

    @Test
    void scoreDifferencingTest() {
        BasicThresholder basicThresholder = new BasicThresholder(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            basicThresholder.setScoreDifferencing(-new Random().nextDouble());
        });
        assertThrows(IllegalArgumentException.class, () -> {
            basicThresholder.setScoreDifferencing(1 + 1e-10 + new Random().nextDouble());
        });
        assertDoesNotThrow(() -> basicThresholder.setScoreDifferencing(new Random().nextDouble()));
    }

    @Test
    void constructorTest() {
        BasicThresholder thresholder = new BasicThresholder(null);
        assertEquals(thresholder.getDeviations().length, 3);

        BasicThresholder thresholder2 = new BasicThresholder(new Deviation[] { new Deviation(0) });
        assertNotNull(thresholder2.getSecondaryDeviation());

        double[] list = new double[] { 1.0, 2.0, 3.0 };
        BasicThresholder basicThresholder = new BasicThresholder(
                DoubleStream.of(list).boxed().collect(Collectors.toList()), 0.01);
        assertEquals(basicThresholder.getPrimaryDeviation().getCount(), 3);
        assertEquals(basicThresholder.getSecondaryDeviation().getCount(), 3);
        assertEquals(basicThresholder.getPrimaryDeviation().getMean(), 2, 1e-10);
        assertEquals(basicThresholder.getSecondaryDeviation().getMean(), 2, 1e-10);
        assertEquals(basicThresholder.getPrimaryDeviation().getDiscount(), 0.01, 1e-10);

        System.out.println(basicThresholder.count);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.updatePrimary(0.0);
        basicThresholder.updatePrimary(0.0);
        System.out.println(basicThresholder.count);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.setScoreDifferencing(0);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.setMinimumScores(5);
        assertTrue(basicThresholder.isDeviationReady());
        basicThresholder.setScoreDifferencing(1.0);
        assertFalse(basicThresholder.isDeviationReady());

        basicThresholder.update(0.0, 0.0);
        basicThresholder.update(0.0, 0.0);
        assertTrue(basicThresholder.isDeviationReady());
        basicThresholder.setScoreDifferencing(0.5);
        assertTrue(basicThresholder.isDeviationReady());
        assertEquals(basicThresholder.intermediateTermFraction(), 0.4, 1e-10);
        basicThresholder.updatePrimary(0.0);
        assertNotEquals(1, basicThresholder.intermediateTermFraction(), 0.0);
        basicThresholder.setMinimumScores(4);
        assertEquals(1, basicThresholder.intermediateTermFraction());

    }

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    void gradeTest(boolean flag) {
        BasicThresholder thresholder = new BasicThresholder(null);
        thresholder.setScoreDifferencing(0.0);
        if (flag) {
            thresholder.setInitialThreshold(0.0);
            thresholder.setAbsoluteThreshold(0.0);
        }
        assertEquals(0, thresholder.threshold());
        assertEquals(0, thresholder.getPrimaryThreshold());
        assertEquals(0, thresholder.getPrimaryGrade(0));
        assertEquals(0, thresholder.getPrimaryThresholdAndGrade(0.0).weight);
        assertEquals(0, thresholder.getPrimaryThresholdAndGrade(1.0).weight);

        assertEquals(thresholder.initialThreshold,
                thresholder.getThresholdAndGrade(0, TransformMethod.NONE, 1, 1).index);
        assertEquals(thresholder.initialThreshold,
                thresholder.getThresholdAndGrade(1.0, TransformMethod.NONE, 1, 1).index);
        thresholder.setCount(12);
        assertTrue(thresholder.isDeviationReady());
        assertEquals(thresholder.getSurpriseIndex(1.0, 0, 2.5, 0), 2);
        assertEquals(thresholder.getPrimaryGrade(0), 0);
        assertEquals(0, thresholder.getPrimaryThresholdAndGrade(0.0).weight);
        assertEquals(0, thresholder.getPrimaryThresholdAndGrade(1.0).weight); // threshold 0
        thresholder.updatePrimary(1.0);
        assertEquals(1.0, thresholder.getPrimaryThresholdAndGrade(2.0).weight);
        thresholder.update(1.0, 1.0);
        thresholder.update(1.0, 0.5);
        assertEquals(0, thresholder.longTermDeviation(TransformMethod.NONE, 1));
        assertEquals(thresholder.getThresholdAndGrade(0, TransformMethod.NONE, 1, 1).weight, 0);
        assertTrue(thresholder.longTermDeviation(TransformMethod.DIFFERENCE, 1) > 0);
        assertTrue(thresholder.longTermDeviation(TransformMethod.NORMALIZE_DIFFERENCE, 1) > 0);
        assertTrue(thresholder.longTermDeviation(TransformMethod.NONE, 2) > 0);
        assertTrue(thresholder.longTermDeviation(TransformMethod.DIFFERENCE, 2) > 0);
        assertTrue(thresholder.longTermDeviation(TransformMethod.NORMALIZE_DIFFERENCE, 2) > 0);
    }

}
