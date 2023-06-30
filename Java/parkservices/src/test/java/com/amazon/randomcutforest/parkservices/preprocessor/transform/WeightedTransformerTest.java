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

package com.amazon.randomcutforest.parkservices.preprocessor.transform;

import static com.amazon.randomcutforest.parkservices.preprocessor.transform.WeightedTransformer.NUMBER_OF_STATS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.parkservices.statistics.Deviation;

public class WeightedTransformerTest {

    @Test
    void constructorTest() {
        assertThrows(IllegalArgumentException.class, () -> new WeightedTransformer(new double[2], new Deviation[5]));
        assertThrows(IllegalArgumentException.class,
                () -> new WeightedTransformer(new double[2], new Deviation[2 * NUMBER_OF_STATS]));
        Deviation[] deviations = new Deviation[NUMBER_OF_STATS];
        for (int i = 0; i < NUMBER_OF_STATS; i++) {
            deviations[i] = new Deviation(0);
        }
        assertDoesNotThrow(() -> new WeightedTransformer(new double[1], deviations));
    }

    @Test
    void updateDeviationsTest() {
        Deviation[] deviations = new Deviation[2 * NUMBER_OF_STATS];
        for (int y = 0; y < deviations.length; y++) {
            deviations[y] = new Deviation(0);
        }
        WeightedTransformer transformer = new WeightedTransformer(new double[2], deviations);
        assertThrows(IllegalArgumentException.class, () -> transformer.updateDeviation(new double[1], new double[1]));
        assertThrows(IllegalArgumentException.class, () -> transformer.updateDeviation(new double[2], new double[1]));
        assertDoesNotThrow(() -> transformer.updateDeviation(new double[2], new double[2]));
    }

    @Test
    void normalizeTest() {
        Deviation[] deviations = new Deviation[2 * NUMBER_OF_STATS];
        for (int y = 0; y < deviations.length; y++) {
            deviations[y] = new Deviation(0);
        }
        WeightedTransformer transformer = new WeightedTransformer(new double[2], deviations);
        assertThrows(IllegalArgumentException.class, () -> transformer.normalize(10, 5, 0, 10));
        assertTrue(transformer.normalize(10, 5, 0.5, 9) == 9);
    }

}
