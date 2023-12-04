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

package com.amazon.randomcutforest.preprocessor.transform;

import static com.amazon.randomcutforest.preprocessor.transform.WeightedTransformer.NUMBER_OF_STATS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.statistics.Deviation;

public class WeightedTransformerTest {

    public void checkTransformer(WeightedTransformer w, double value, double another) {
        w.setWeights(new double[1]);
        float[] test = new float[] { 1.0f };
        w.invertInPlace(test, new double[] { 2.0 });
        assertEquals(test[0], value, 1e-6);
        assertEquals(w.getScale()[0], 0, 1e-6);
        RangeVector r = new RangeVector(1);
        r.shift(0, 10);
        assertEquals(r.values[0], 10, 1e-6);
        assertEquals(r.upper[0], 10, 1e-6);
        assertEquals(r.lower[0], 10, 1e-6);
        assertThrows(IllegalArgumentException.class,
                () -> w.invertForecastRange(r, 1, new double[] { 1.0 }, new double[0]));
        w.invertForecastRange(r, 1, new double[] { 1.0 }, new double[1]);
        assertEquals(r.values[0], another, 1e-6);
        assertEquals(r.upper[0], another, 1e-6);
        assertEquals(r.lower[0], another, 1e-6);
    }

    @Test
    void constructorTest() {
        assertThrows(IllegalArgumentException.class, () -> new WeightedTransformer(new double[2], new Deviation[5]));
        assertThrows(IllegalArgumentException.class,
                () -> new WeightedTransformer(new double[2], new Deviation[2 * NUMBER_OF_STATS]));
        Deviation[] deviations = new Deviation[NUMBER_OF_STATS];
        for (int i = 0; i < NUMBER_OF_STATS; i++) {
            deviations[i] = new Deviation(0);
        }
        WeightedTransformer w = new WeightedTransformer(new double[1], deviations);
        assertThrows(IllegalArgumentException.class, () -> w.setWeights(new double[2]));
        checkTransformer(w, 0, 0);
        checkTransformer(new NormalizedDifferenceTransformer(new double[1], deviations), 2.0, 1.0);
        assertThrows(IllegalArgumentException.class,
                () -> new NormalizedDifferenceTransformer(new double[1], deviations).invertInPlace(new float[1],
                        new double[2]));
        checkTransformer(new DifferenceTransformer(new double[1], deviations), 2.0, 1.0);
        assertThrows(IllegalArgumentException.class,
                () -> new DifferenceTransformer(new double[1], deviations).invertInPlace(new float[1], new double[2]));
    }

    @Test
    void updateDeviationsTest() {
        Deviation[] deviations = new Deviation[2 * NUMBER_OF_STATS];
        for (int y = 0; y < deviations.length; y++) {
            deviations[y] = new Deviation(0);
        }
        WeightedTransformer transformer = new WeightedTransformer(new double[2], deviations);
        assertThrows(IllegalArgumentException.class,
                () -> transformer.updateDeviation(new double[1], new double[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> transformer.updateDeviation(new double[2], new double[1], null));
        assertDoesNotThrow(() -> transformer.updateDeviation(new double[2], new double[2], null));
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
        assertTrue(transformer.normalize(-10, -5, 0.5, 9) == -9);
    }

}
