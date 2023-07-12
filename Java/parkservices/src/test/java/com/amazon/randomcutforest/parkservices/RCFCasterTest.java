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
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
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
    public void configTest() {
        RCFCaster.Builder builder = new RCFCaster.Builder().dimensions(1).shingleSize(1).forecastHorizon(1);
        RCFCaster caster = builder.build();
        assertThrows(IllegalArgumentException.class,
                () -> caster.processSequentially(new double[][] { new double[0] }));
        assertThrows(IllegalArgumentException.class, () -> caster.process(new double[1], 0L, new int[1]));
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

        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 1, 0.1, 1, new float[2], new float[6], new float[1], null));
        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 1, 0.1, 1, null, null, new float[2], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 1, 0.1, 1, new float[2], null, new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 1, 0.1, 1, null, new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 0, 1, 0.1, 2, new float[2], new float[6], new float[2], null));

        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 2, 1, 0.1, 1, new float[2], new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.1, 1, new float[2], new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 0, new float[2], new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.1, 1, new float[2], new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 3, new float[2], new float[6], new float[3], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.6, 1, new float[2], new float[6], new float[1], null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0.1, 2, new float[2], new float[6], new float[2], null));
    }

    @Test
    public void testCalibrate() {
        ErrorHandler e = new ErrorHandler(new RCFCaster.Builder().errorHorizon(2).forecastHorizon(2).dimensions(2));
        assertThrows(IllegalArgumentException.class, () -> e.calibrate(Calibration.SIMPLE, new RangeVector(5)));
        RangeVector r = new RangeVector(4);
        e.sequenceIndex = 5;
        e.lastDeviations = new float[] { 1.0f, 1.3f };
        float v = new Random().nextFloat();
        r.shift(0, v);
        e.calibrate(Calibration.SIMPLE, new RangeVector(r));
        assertEquals(r.values[0], v);
        e.calibrate(Calibration.NONE, r);
        assertEquals(r.values[0], v);
        assertEquals(r.upper[0], v);
        assertEquals(r.values[1], 0);
        e.lastDeviations = new float[] { v + 1.0f, 1.3f };
        e.calibrate(Calibration.MINIMAL, r);
        assertEquals(r.values[0], v);
        assertEquals(r.values[1], 0);
        assertEquals(r.upper[0], v + 1.3 * (v + 1), 1e-6f);
        assertEquals(r.lower[0], v - 1.3 * (v + 1), 1e-6f);
        e.sequenceIndex = 10000;
        e.errorHorizon = 1000;
        RangeVector newR = new RangeVector(4);
        newR.shift(0, v);
        e.errorDistribution.shift(0, 2 * v);
        e.calibrate(Calibration.SIMPLE, newR);
        assertEquals(newR.values[0], 3 * v, 1e-6f);
        assertEquals(newR.values[1], 0);
        assertThrows(IllegalArgumentException.class, () -> e.adjustMinimal(0, new RangeVector(10), new RangeVector(9)));
        assertThrows(IllegalArgumentException.class,
                () -> e.adjustMinimal(10, new RangeVector(10), new RangeVector(10)));
        assertThrows(IllegalArgumentException.class,
                () -> e.adjustMinimal(-1, new RangeVector(10), new RangeVector(10)));
        assertThrows(IllegalArgumentException.class, () -> e.interpolatedMedian(new double[6], 25));
        assertThrows(IllegalArgumentException.class, () -> e.interpolatedMedian(null, 25));
        assertDoesNotThrow(() -> e.interpolatedMedian(new double[25], 25));
        assertThrows(IllegalArgumentException.class, () -> e.adjust(0, new RangeVector(9), new RangeVector(10)));
        assertThrows(IllegalArgumentException.class, () -> e.adjust(9, new RangeVector(9), new RangeVector(9)));
        assertThrows(IllegalArgumentException.class, () -> e.adjust(-1, new RangeVector(9), new RangeVector(9)));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 2, new float[2], new float[6], new float[1], null));
    }

    @ParameterizedTest
    @EnumSource(Calibration.class)
    void testRCFCast(Calibration calibration) {
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 2 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;
        int forecastHorizon = 5; // speeding up
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
                .centerOfMassEnabled(true).storeSequenceIndexesEnabled(true) // neither is relevant
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125).build();
        RCFCaster shadow = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125)
                .boundingBoxCacheFraction(0).build();
        RCFCaster secondShadow = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125)
                .boundingBoxCacheFraction(0).build();
        RCFCaster thirdShadow = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125)
                .boundingBoxCacheFraction(1.0).build();

        // testing scoring strategies
        caster.setScoringStrategy(ScoringStrategy.MULTI_MODE);
        shadow.setScoringStrategy(ScoringStrategy.MULTI_MODE);
        // ensuring/testing that the parameters are the same; otherwise the
        // grades/scores cannot
        // be the same
        caster.setLowerThreshold(1.1);
        shadow.setLowerThreshold(1.1);
        secondShadow.setLowerThreshold(1.1);
        thirdShadow.setLowerThreshold(1.1);
        caster.setInitialThreshold(2.0);
        shadow.setInitialThreshold(2.0);
        secondShadow.setInitialThreshold(2.0);
        thirdShadow.setInitialThreshold(2.0);
        caster.setScoreDifferencing(0.4);
        shadow.setScoreDifferencing(0.4);
        secondShadow.setScoreDifferencing(0.4);
        thirdShadow.setScoreDifferencing(0.4);

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
                            double[] another = caster.errorHandler.getErrorVector(len, (i + 1), k, pos,
                                    RCFCaster.alternateError);
                            // smape; calibration may increase errors
                            assertTrue(calibration != Calibration.NONE || Arrays.stream(another).sum() < 2 * len);
                        }
                    }
                }
                float[] intervalPrecision = shadow.errorHandler.getIntervalPrecision();
                for (float y : intervalPrecision) {
                    assertTrue(0 <= y && y <= 1.0);
                }
                assertArrayEquals(intervalPrecision, result.getIntervalPrecision(), 1e-6f);
                float[] test = new float[forecastHorizon * baseDimensions];
                assertArrayEquals(caster.errorHandler.getAdders().values, test, 1e-6f);
                Arrays.fill(test, 1);
                assertArrayEquals(caster.errorHandler.getMultipliers().values, test, 1e-6f);
            }
        }

        // 0 length arrays do not change state
        secondShadow.processSequentially(new double[0][]);
        List<AnomalyDescriptor> firstList = secondShadow.processSequentially(dataWithKeys.data);
        List<AnomalyDescriptor> thirdList = thirdShadow.processSequentially(dataWithKeys.data);
        // null does not change state
        thirdShadow.processSequentially(null);

        // calibration fails
        assertThrows(IllegalArgumentException.class, () -> caster.extrapolate(forecastHorizon - 1));
        assertThrows(IllegalArgumentException.class, () -> caster.extrapolate(forecastHorizon + 1));

        TimedRangeVector forecast1 = caster.extrapolate(forecastHorizon);
        TimedRangeVector forecast2 = shadow.extrapolate(forecastHorizon);

        TimedRangeVector forecast3 = secondShadow.extrapolate(forecastHorizon);
        TimedRangeVector forecast4 = thirdShadow.extrapolate(forecastHorizon);
        assertArrayEquals(forecast1.rangeVector.values, forecast2.rangeVector.values, 1e-6f);

        assertArrayEquals(forecast3.rangeVector.values, forecast4.rangeVector.values, 1e-6f);
        // the order of floating point operations now vary

        for (int i = 0; i < forecast1.rangeVector.values.length; i++) {
            assertTrue(Math.abs(forecast1.rangeVector.values[i] - forecast3.rangeVector.values[i]) < 1e-4
                    * (1 + Math.abs(forecast1.rangeVector.values[i])));
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

            TimedRangeVector timedShadowForecast = shadow.extrapolate(forecastHorizon);

            assertArrayEquals(timedShadowForecast.timeStamps, result.getTimedForecast().timeStamps);
            assertArrayEquals(timedShadowForecast.upperTimeStamps, result.getTimedForecast().upperTimeStamps);
            assertArrayEquals(timedShadowForecast.lowerTimeStamps, result.getTimedForecast().lowerTimeStamps);

            // first check idempotence -- forecasts are state dependent only
            // for ThresholdedRCF
            TimedRangeVector newShadow = shadow.extrapolate(forecastHorizon);
            assertArrayEquals(newShadow.rangeVector.values, timedShadowForecast.rangeVector.values, 1e-6f);
            assertArrayEquals(newShadow.rangeVector.upper, timedShadowForecast.rangeVector.upper, 1e-6f);
            assertArrayEquals(newShadow.rangeVector.lower, timedShadowForecast.rangeVector.lower, 1e-6f);
            assertArrayEquals(newShadow.timeStamps, timedShadowForecast.timeStamps);
            assertArrayEquals(newShadow.upperTimeStamps, timedShadowForecast.upperTimeStamps);
            assertArrayEquals(newShadow.lowerTimeStamps, timedShadowForecast.lowerTimeStamps);

            // extrapolate is idempotent for RCF casters
            TimedRangeVector newVector = caster.extrapolate(forecastHorizon);
            assertArrayEquals(newVector.rangeVector.values, result.getTimedForecast().rangeVector.values, 1e-6f);
            assertArrayEquals(newVector.rangeVector.upper, result.getTimedForecast().rangeVector.upper, 1e-6f);
            assertArrayEquals(newVector.rangeVector.lower, result.getTimedForecast().rangeVector.lower, 1e-6f);
            assertArrayEquals(newVector.timeStamps, result.getTimedForecast().timeStamps);
            assertArrayEquals(newVector.upperTimeStamps, result.getTimedForecast().upperTimeStamps);
            assertArrayEquals(newVector.lowerTimeStamps, result.getTimedForecast().lowerTimeStamps);

            // only difference between RCFCaster and ThresholdedRCF is calibration
            caster.calibrate(calibration, timedShadowForecast.rangeVector);
            assertArrayEquals(timedShadowForecast.rangeVector.values, result.getTimedForecast().rangeVector.values,
                    1e-6f);
            assertArrayEquals(timedShadowForecast.rangeVector.upper, result.getTimedForecast().rangeVector.upper,
                    1e-6f);
            assertArrayEquals(timedShadowForecast.rangeVector.lower, result.getTimedForecast().rangeVector.lower,
                    1e-6f);

        }
    }

}
