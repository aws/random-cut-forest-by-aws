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

import static com.amazon.randomcutforest.parkservices.ErrorHandler.MAX_ERROR_HORIZON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class RCFCasterTest {

    @Test
    public void constructorTest() {
        RCFCaster.Builder builder = new RCFCaster.Builder();
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.forecastHorizon(-1);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.forecastHorizon(2).shingleSize(0);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.shingleSize(1).dimensions(1);
        assertDoesNotThrow(builder::build);
        builder.internalShinglingEnabled(false);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.internalShinglingEnabled(true);
        assertDoesNotThrow(builder::build);
        builder.forestMode(ForestMode.STREAMING_IMPUTE);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.forestMode(ForestMode.TIME_AUGMENTED);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    public void errorHandlerConstructorTest() {
        RCFCaster.Builder builder = new RCFCaster.Builder();
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
        builder.errorHorizon(MAX_ERROR_HORIZON + 1);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));

        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 1, 0.1, 1, new float[2], new float[6], null));
        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 1, 0.1, 1, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 1, 0.1, 1, new float[2], null, null));
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 1, 0.1, 1, null, new float[6], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 2, 1, 0.1, 1, new float[2], new float[6], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.1, 1, new float[2], new float[6], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, -0.1, 1, new float[2], new float[6], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.5, 1, new float[2], new float[6], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.1, 2, new float[2], new float[6], null));
    }

    @Test
    public void testCalibrate() {
        ErrorHandler e = new ErrorHandler(new RCFCaster.Builder().errorHorizon(2).forecastHorizon(2).dimensions(2));
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(Calibration.SIMPLE, new double[1], new RangeVector(4)));
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(Calibration.SIMPLE, new double[2], new RangeVector(5)));
        assertThrows(IllegalArgumentException.class, () -> e.calibrate(Calibration.SIMPLE, null, new RangeVector(5)));
    }

    @ParameterizedTest
    @EnumSource(Calibration.class)
    void testRCFCast(Calibration calibration) {
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 10 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;
        int forecastHorizon = 15;
        int shingleSize = 10;
        int outputAfter = 32;
        int errorHorizon = 256;

        long seed = new Random().nextLong();

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                50, 5, seed, baseDimensions, false);

        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NORMALIZE;
        RCFCaster caster = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125).build();
        RCFCaster shadow = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125)
                .boundingBoxCacheFraction(0).build();

        // ensuring that the parameters are the same; otherwise the grades/scores cannot
        // be the same
        caster.setLowerThreshold(1.1);
        shadow.setLowerThreshold(1.1);

        assert (caster.errorHandler.errorHorizon == errorHorizon);
        assert (caster.errorHorizon == errorHorizon);
        for (int j = 0; j < dataWithKeys.data.length; j++) {
            ForecastDescriptor result = caster.process(dataWithKeys.data[j], 0L);
            ForecastDescriptor shadowResult = shadow.process(dataWithKeys.data[j], 0L);
            assertEquals(result.getRCFScore(), shadowResult.getRCFScore(), 1e-6);
            assertArrayEquals(shadowResult.getTimedForecast().rangeVector.values,
                    result.getTimedForecast().rangeVector.values, 1e-6f);
            assertArrayEquals(shadowResult.getTimedForecast().rangeVector.upper,
                    result.getTimedForecast().rangeVector.upper, 1e-6f);
            assertArrayEquals(shadowResult.getTimedForecast().rangeVector.lower,
                    result.getTimedForecast().rangeVector.lower, 1e-6f);

            int sequenceIndex = caster.errorHandler.sequenceIndex;
            if (caster.forest.isOutputReady()) {
                float[] meanArray = caster.errorHandler.getErrorMean();
                for (int i = 0; i < forecastHorizon; i++) {
                    int len = (sequenceIndex > errorHorizon + i + 1) ? errorHorizon : sequenceIndex - i - 1;
                    if (len > 0) {
                        for (int k = 0; k < baseDimensions; k++) {
                            int pos = i * baseDimensions + k;
                            double[] array = caster.errorHandler.getErrorVector(len, (i + 1), k, pos,
                                    RCFCaster.defaultError);
                            double mean = Arrays.stream(array).sum() / len;
                            assertEquals(meanArray[pos], mean, (1 + Math.abs(mean)) * 1e-4);
                        }
                    }
                }
                float[] intervalPrecision = shadow.errorHandler.getIntervalPrecision();
                for (float y : intervalPrecision) {
                    assertTrue(0 <= y && y <= 1.0);
                }
                assertArrayEquals(intervalPrecision, result.getIntervalPrecision());
                float[] test = new float[forecastHorizon * baseDimensions];
                assertArrayEquals(caster.errorHandler.getAdders().values, test, 1e-6f);
                Arrays.fill(test, 1);
                assertArrayEquals(caster.errorHandler.getMultipliers().values, test, 1e-6f);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Calibration.class)
    void testRCFCastThresholdedRCF(Calibration calibration) {
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;
        int forecastHorizon = 15;
        int shingleSize = 10;
        int outputAfter = 32;
        int errorHorizon = 256;

        long seed = -1908593580679997334L;
        new Random().nextLong();

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                50, 5, seed, baseDimensions, false);

        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NORMALIZE;

        RCFCaster caster = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125).build();
        ThresholdedRandomCutForest shadow = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(seed + 1).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).initialAcceptFraction(0.125).build();

        // ensuring that the parameters are the same; otherwise the grades/scores cannot
        // be the same
        // weighTime has to be 0
        caster.setLowerThreshold(1.1);
        shadow.setLowerThreshold(1.1);

        assertTrue(caster.errorHandler.errorHorizon == errorHorizon);
        assertTrue(caster.errorHorizon == errorHorizon);

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            ForecastDescriptor result = caster.process(dataWithKeys.data[j], 0L);
            AnomalyDescriptor shadowResult = shadow.process(dataWithKeys.data[j], 0L);
            assertEquals(result.getRCFScore(), shadowResult.getRCFScore(), 1e-6f);

            RangeVector shadowForecast = shadow.extrapolate(forecastHorizon).rangeVector;
            // all other callibration may alter the forecast
            if (calibration == Calibration.NONE) {
                assertArrayEquals(shadowForecast.values, result.getTimedForecast().rangeVector.values, 1e-6f);
            }

            if (j > outputAfter) {
                // but we can invoke the same calibration
                caster.calibrate(calibration, shadowResult.getPostDeviations(), shadowForecast);
                assertArrayEquals(shadowForecast.values, result.getTimedForecast().rangeVector.values, 1e-6f);
                assertArrayEquals(shadowForecast.upper, result.getTimedForecast().rangeVector.upper, 1e-6f);
                assertArrayEquals(shadowForecast.lower, result.getTimedForecast().rangeVector.lower, 1e-6f);
            }
        }
    }

}
