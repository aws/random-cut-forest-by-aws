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

package com.amazon.randomcutforest.parkservices.statistics;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;

public class StatisticsTest {

    @Test
    void constructorTest() {
        assertThrows(IllegalArgumentException.class, () -> new Deviation(-1));

        assertThrows(IllegalArgumentException.class, () -> new Deviation(2));

        assertDoesNotThrow(() -> new Deviation(new Random().nextDouble()));
    }

    @Test
    void getMeanTest() {
        Deviation deviation = new Deviation();
        assertEquals(deviation.getMean(), 0);
        assertTrue(deviation.isEmpty());
        deviation.update(-0);
        assertEquals(1, deviation.count);
        assertEquals(deviation.getMean(), 0);
    }

}
