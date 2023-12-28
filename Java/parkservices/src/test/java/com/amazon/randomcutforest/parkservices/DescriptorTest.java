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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

public class DescriptorTest {

    int dimensions;
    int horizon;
    private ForecastDescriptor forecastDescriptor;

    @BeforeEach
    public void setUp() {
        dimensions = 4;
        horizon = 2;
        forecastDescriptor = new ForecastDescriptor(new double[] { 2.0, 3.0 }, 0L, 7);

    }

    @Test
    public void testSet() {
        assertThrows(IllegalArgumentException.class,
                () -> forecastDescriptor.setObservedErrorDistribution(new RangeVector(15)));
        assertDoesNotThrow(() -> forecastDescriptor.setObservedErrorDistribution(new RangeVector(14)));
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setErrorRMSE(new DiVector(13)));
        assertDoesNotThrow(() -> forecastDescriptor.setErrorRMSE(new DiVector(14)));

        assertFalse(forecastDescriptor.isExpectedValuesPresent());
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setExpectedValues(2, new double[2], 1.0));
        forecastDescriptor.setExpectedValues(0, new double[2], 1.0);
        assertTrue(forecastDescriptor.isExpectedValuesPresent());
        assertNull(forecastDescriptor.getLastExpectedRCFPoint());
        assertArrayEquals(forecastDescriptor.getExpectedValuesList()[0], new double[2]);
        forecastDescriptor.setExpectedValues(0, new double[] { -1.0, -1.0 }, 0.5);
        assertArrayEquals(forecastDescriptor.getExpectedValuesList()[0], new double[] { -1.0, -1.0 });
        assertNull(forecastDescriptor.getMissingValues());
        forecastDescriptor.setMissingValues(null);
        assertNull(forecastDescriptor.getMissingValues());
        forecastDescriptor.setMissingValues(new int[] { 17 });
        assertArrayEquals(forecastDescriptor.getMissingValues(), new int[] { 17 });
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(0, null));
        forecastDescriptor.setNumberOfNewImputes(1);
        forecastDescriptor.setInputLength(1);
        forecastDescriptor.setShingleSize(1);
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(0, null));
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(0, new double[3]));
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(0, new double[1]));
        forecastDescriptor.setShingleSize(2);
        forecastDescriptor.setImputedPoints(null); // reset
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(-1, new double[1]));
        assertDoesNotThrow(() -> forecastDescriptor.setImputedPoint(0, new double[1]));
        // cannot set twice
        assertThrows(IllegalArgumentException.class, () -> forecastDescriptor.setImputedPoint(0, new double[1]));
    }

}
