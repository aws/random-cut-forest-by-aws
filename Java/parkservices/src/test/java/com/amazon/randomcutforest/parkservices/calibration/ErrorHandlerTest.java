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

package com.amazon.randomcutforest.parkservices.calibration;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.parkservices.config.Calibration;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.statistics.Deviation;

public class ErrorHandlerTest {

    @Test
    public void errorHandlerConstructorTest() {
        ErrorHandler.Builder builder = new ErrorHandler.Builder();
        // builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
        // .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
        // .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
        // .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
        // .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.errorHorizon(1).forecastHorizon(2);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.errorHorizon(2).forecastHorizon(2);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.dimensions(1);
        assertDoesNotThrow(() -> new ErrorHandler(builder));
        builder.errorHorizon(10000);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
    }

    @Test
    public void testCalibrate() {
        ErrorHandler e = ErrorHandler.builder().errorHorizon(2).forecastHorizon(2).dimensions(2).build();
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(new double[2], Calibration.SIMPLE, new RangeVector(5)));
        RangeVector r = new RangeVector(4);
        e.sequenceIndex = 5;
        e.lastDataDeviations = new float[] { 1.0f, 1.3f };
        float v = new Random().nextFloat();
        r.shift(0, v);
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(new double[1], Calibration.SIMPLE, new RangeVector(r)));
        e.calibrate(new double[2], Calibration.SIMPLE, new RangeVector(r));
        assertEquals(r.values[0], v);
        e.calibrate(new double[2], Calibration.NONE, r);
        assertEquals(r.values[0], v);
        assertEquals(r.upper[0], v);
        assertEquals(r.values[1], 0);
        e.lastDataDeviations = new float[] { v + 1.0f, 1.3f };
        e.calibrate(new double[2], Calibration.MINIMAL, r);
        assertEquals(r.values[0], v);
        assertEquals(r.values[1], 0);
    }

    @Test
    public void testSerializedConstructor() {
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(0, 0, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(0, 1, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0, 1, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0, 1, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, null, null, new Deviation[2], null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, null, new double[1], new Deviation[3], null, null));
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, new float[0],
                new double[2], new Deviation[3], null, null));
        Deviation[] deviations = new Deviation[3];
        deviations[0] = deviations[1] = deviations[2] = new Deviation(0);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 0, 0.1, 1, new float[2], new float[1],
                new double[2], deviations, null, null));
        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 0, 0.1, 1, new float[0], new float[1], new double[2],
                deviations, null, null));
    }

}
