/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.anomalydetection;

import java.util.function.BiFunction;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DynamicScoreVisitorTest {
    @Test
    public void testScoringMethods() {
        BiFunction<Double, Double, Double> scoreSeen = (x, y) -> (x + y) / 2;
        BiFunction<Double, Double, Double> scoreUneen = (x, y) -> 0.75 * x + 0.25 * y;
        BiFunction<Double, Double, Double> damp = (x, y) -> Math.sqrt(x * y);
        DynamicScoreVisitor visitor = new DynamicScoreVisitor(new double[] {1.1, -2.2}, 100,  2,
                scoreSeen, scoreUneen, damp);

        int x = 9;
        int y = 4;
        assertEquals((x + y) / 2.0, visitor.scoreSeen(x, y));
        assertEquals(0.75 * x + 0.25 * y, visitor.scoreUnseen(x, y));
        assertEquals(Math.sqrt(x * y), visitor.damp(x, y));
    }
}
