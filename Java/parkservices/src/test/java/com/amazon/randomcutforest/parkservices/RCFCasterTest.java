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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.config.Calibration;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
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
        builder.shingleSize(1).dimensions(1).scoreDifferencing(0);
        assertDoesNotThrow(builder::build);
        // unlikely to succeed; independent random number generator
        assertNotEquals(builder.getRandom().nextInt(), builder.getRandom().nextInt());
        builder.randomSeed(10);
        assertEquals(builder.getRandom().nextInt(), builder.getRandom().nextInt());
        builder.internalShinglingEnabled(false);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.internalShinglingEnabled(true);
        assertDoesNotThrow(builder::build);
        builder.forestMode(ForestMode.STREAMING_IMPUTE);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.forestMode(ForestMode.TIME_AUGMENTED);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.forestMode(ForestMode.STANDARD);
        builder.upperLimit(new float[] {});
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.upperLimit(new float[] { 1.0f });
        builder.lowerLimit(new float[] {});
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.lowerLimit(new float[] { 2.0f });
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.lowerLimit(new float[] { 0.0f });
        builder.parallelExecutionEnabled(true).threadPoolSize(2).zFactor(2.0);
        assertDoesNotThrow(builder::build);
        builder.startNormalization(-1);
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.startNormalization(200).outputAfter(1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    public void configTest() {
        RCFCaster.Builder builder = new RCFCaster.Builder().dimensions(1).shingleSize(1).forecastHorizon(1);
        RCFCaster caster = builder.build();
        caster.setLowerLimit(null);
        caster.setUpperLimit(null);
        assertThrows(IllegalArgumentException.class, () -> caster.setUpperLimit(new float[] { 0, 0 }));
        assertThrows(IllegalArgumentException.class, () -> caster.setLowerLimit(new float[] { 0, 0 }));
        assertDoesNotThrow(() -> caster.setUpperLimit(new float[] { 0 }));
        assertThrows(IllegalArgumentException.class, () -> caster.setLowerLimit(new float[] { 1 }));
        assertThrows(IllegalArgumentException.class,
                () -> caster.processSequentially(new double[][] { new double[0] }));
        assertThrows(IllegalArgumentException.class, () -> caster.process(new double[1], 0L, new int[1]));
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
        secondShadow.setScoringStrategy(ScoringStrategy.MULTI_MODE);
        thirdShadow.setScoringStrategy(ScoringStrategy.MULTI_MODE);
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

        assert (caster.errorHandler.getErrorHorizon() == errorHorizon);
        assert (caster.errorHorizon == errorHorizon);

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            ForecastDescriptor result = caster.process(dataWithKeys.data[j], 0L);
            ForecastDescriptor shadowResult = shadow.process(dataWithKeys.data[j], 0L);
            assertEquals(result.getRCFScore(), shadowResult.getRCFScore(), 1e-6);
            /*
             * assertArrayEquals(shadowResult.getTimedForecast().rangeVector.values,
             * result.getTimedForecast().rangeVector.values, 1e-1f);
             * assertArrayEquals(shadowResult.getTimedForecast().rangeVector.upper,
             * result.getTimedForecast().rangeVector.upper, 1e-1f);
             * assertArrayEquals(shadowResult.getTimedForecast().rangeVector.lower,
             * result.getTimedForecast().rangeVector.lower, 1e-1f);
             */
            int sequenceIndex = caster.errorHandler.getSequenceIndex();
            if (caster.forest.isOutputReady()) {
                float[] meanArray = caster.errorHandler.getErrorMean();
                float[] intervalPrecision = shadow.errorHandler.getIntervalPrecision();
                for (float y : intervalPrecision) {
                    assertTrue(0 <= y && y <= 1.0);
                }
                // assertArrayEquals(intervalPrecision, result.getIntervalPrecision(), 1e-6f);
            }
        }

        // 0 length arrays do not change state
        secondShadow.processSequentially(new double[0][]);
        List<AnomalyDescriptor> firstList = secondShadow.processSequentially(dataWithKeys.data);
        List<AnomalyDescriptor> thirdList = thirdShadow.processSequentially(dataWithKeys.data);
        // null does not change state
        thirdShadow.processSequentially(null);

        if (calibration != Calibration.NONE) {
            // calibration fails
            assertThrows(IllegalArgumentException.class, () -> caster.extrapolate(forecastHorizon - 1));
            assertThrows(IllegalArgumentException.class, () -> caster.extrapolate(forecastHorizon + 1));
        }

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

        assertTrue(caster.errorHandler.getErrorHorizon() == errorHorizon);
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
            caster.calibrate(dataWithKeys.data[j], calibration, timedShadowForecast.rangeVector);
            assertArrayEquals(timedShadowForecast.rangeVector.values, result.getTimedForecast().rangeVector.values,
                    1e-6f);
            assertArrayEquals(timedShadowForecast.rangeVector.upper, result.getTimedForecast().rangeVector.upper,
                    1e-6f);
            assertArrayEquals(timedShadowForecast.rangeVector.lower, result.getTimedForecast().rangeVector.lower,
                    1e-6f);

        }
    }

}
