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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.junit.jupiter.api.Test;

public class BasicThresholderTest {

    @Test
    void horizonTest() {
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
        double[] list = new double[] { 1.0, 2.0, 3.0 };
        BasicThresholder basicThresholder = new BasicThresholder(
                DoubleStream.of(list).boxed().collect(Collectors.toList()), 0.01);
        assertEquals(basicThresholder.getPrimaryDeviation().getCount(), 3);
        assertEquals(basicThresholder.getSecondaryDeviation().getCount(), 3);
        assertEquals(basicThresholder.getPrimaryDeviation().getMean(), 2, 1e-10);
        assertEquals(basicThresholder.getSecondaryDeviation().getMean(), 2, 1e-10);
        assertEquals(basicThresholder.getPrimaryDeviation().getDiscount(), 0.01, 1e-10);

        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.updatePrimary(0.0);
        basicThresholder.updatePrimary(0.0);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.setScoreDifferencing(0);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.setMinimumScores(5);
        assertFalse(basicThresholder.isDeviationReady());
        basicThresholder.setScoreDifferencing(1.0);
        assertTrue(basicThresholder.isDeviationReady());

        basicThresholder.updatePrimary(0.0);
        basicThresholder.updatePrimary(0.0);
        assertEquals(basicThresholder.intermediateTermFraction(), 0.4, 1e-10);
        basicThresholder.updatePrimary(0.0);
        assertFalse(basicThresholder.intermediateTermFraction() == 1);
        basicThresholder.setMinimumScores(4);
        assertTrue(basicThresholder.intermediateTermFraction() == 1);
    }

}
