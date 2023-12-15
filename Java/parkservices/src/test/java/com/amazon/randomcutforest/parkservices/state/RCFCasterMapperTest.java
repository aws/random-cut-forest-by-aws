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

package com.amazon.randomcutforest.parkservices.state;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;
import com.amazon.randomcutforest.parkservices.RCFCaster;
import com.amazon.randomcutforest.parkservices.config.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

public class RCFCasterMapperTest {

    @ParameterizedTest
    @CsvSource({ "SIMPLE,1", "MINIMAL,1", "NONE,1", "SIMPLE,2", "MINIMAL,2", "NONE,2" })
    public void testRoundTripStandardShingleSizeEight(String calibrationString, int inputLength) {
        int shingleSize = 8;
        int dimensions = inputLength * shingleSize;
        int forecastHorizon = shingleSize * 3;
        for (int trials = 0; trials < 1; trials++) {

            long seed = new Random().nextLong();
            System.out.println(" seed " + seed);
            // note shingleSize == 8
            RCFCaster first = RCFCaster.builder().dimensions(dimensions).randomSeed(seed).internalShinglingEnabled(true)
                    .anomalyRate(0.01).shingleSize(shingleSize).calibration(Calibration.MINIMAL)
                    .forecastHorizon(forecastHorizon).calibration(Calibration.valueOf(calibrationString))
                    .transformMethod(TransformMethod.NORMALIZE).build();

            Random r = new Random(seed);
            for (int i = 0; i < 2000 + r.nextInt(1000); i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
                first.process(point, 0L);
            }

            // serialize + deserialize
            RCFCasterMapper mapper = new RCFCasterMapper();
            RCFCaster second = mapper.toModel(mapper.toState(first));
            assertArrayEquals(first.getErrorHandler().getIntervalPrecision(),
                    second.getErrorHandler().getIntervalPrecision(), 1e-6f);
            assertArrayEquals(first.getErrorHandler().getErrorRMSE().high, second.getErrorHandler().getErrorRMSE().high,
                    1e-6f);
            assertArrayEquals(first.getErrorHandler().getErrorRMSE().low, second.getErrorHandler().getErrorRMSE().low,
                    1e-6f);
            assertArrayEquals(first.getErrorHandler().getErrorDistribution().values,
                    second.getErrorHandler().getErrorDistribution().values, 1e-6f);
            assertArrayEquals(first.getErrorHandler().getErrorDistribution().upper,
                    second.getErrorHandler().getErrorDistribution().upper, 1e-6f);
            assertArrayEquals(first.getErrorHandler().getErrorDistribution().lower,
                    second.getErrorHandler().getErrorDistribution().lower, 1e-6f);
            // update re-instantiated forest
            for (int i = 0; i < 100; i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
                ForecastDescriptor firstResult = first.process(point, 0L);
                ForecastDescriptor secondResult = second.process(point, 0L);
                assertEquals(firstResult.getDataConfidence(), secondResult.getDataConfidence(), 1e-10);
                verifyForecast(firstResult, secondResult, inputLength);
            }
        }
    }

    void verifyForecast(ForecastDescriptor firstResult, ForecastDescriptor secondResult, int inputLength) {
        RangeVector firstForecast = firstResult.getTimedForecast().rangeVector;
        RangeVector secondForecast = secondResult.getTimedForecast().rangeVector;
        assertArrayEquals(firstForecast.values, secondForecast.values, 1e-6f);
        assertArrayEquals(firstForecast.upper, secondForecast.upper, 1e-6f);
        assertArrayEquals(firstForecast.lower, secondForecast.lower, 1e-6f);

        float[] firstErrorP50 = firstResult.getObservedErrorDistribution().values;
        float[] secondErrorP50 = secondResult.getObservedErrorDistribution().values;
        assertArrayEquals(firstErrorP50, secondErrorP50, 1e-6f);

        float[] firstUpperError = firstResult.getObservedErrorDistribution().upper;
        float[] secondUpperError = secondResult.getObservedErrorDistribution().upper;
        assertArrayEquals(firstUpperError, secondUpperError, 1e-6f);

        float[] firstLowerError = firstResult.getObservedErrorDistribution().lower;
        float[] secondLowerError = secondResult.getObservedErrorDistribution().lower;
        assertArrayEquals(firstLowerError, secondLowerError, 1e-6f);

        DiVector firstRmse = firstResult.getErrorRMSE();
        DiVector secondRmse = secondResult.getErrorRMSE();
        assertArrayEquals(firstRmse.high, secondRmse.high, 1e-6);
        assertArrayEquals(firstRmse.low, secondRmse.low, 1e-6);

        assertArrayEquals(firstResult.getErrorMean(), secondResult.getErrorMean(), 1e-6f);
        assertArrayEquals(firstResult.getIntervalPrecision(), secondResult.getIntervalPrecision(), 1e-6f);
    }

    @ParameterizedTest
    @CsvSource({ "SIMPLE,1", "MINIMAL,1", "NONE,1", "SIMPLE,2", "MINIMAL,2", "NONE,2" })
    public void testNotFullyInitialized(String calibrationString, int inputLength) {
        int shingleSize = 8;
        int dimensions = inputLength * shingleSize;
        int forecastHorizon = shingleSize * 3;
        int outputAfter = 32;
        for (int trials = 0; trials < 10; trials++) {

            long seed = new Random().nextLong();
            System.out.println(" seed " + seed);

            // note shingleSize == 8
            RCFCaster first = RCFCaster.builder().dimensions(dimensions).randomSeed(seed).internalShinglingEnabled(true)
                    .anomalyRate(0.01).shingleSize(shingleSize).calibration(Calibration.valueOf(calibrationString))
                    .forecastHorizon(forecastHorizon).transformMethod(TransformMethod.NORMALIZE)
                    .outputAfter(outputAfter).build();

            Random r = new Random();
            for (int i = 0; i < new Random().nextInt(outputAfter); i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
                RCFCasterMapper mapper = new RCFCasterMapper();
                RCFCaster shadow = mapper.toModel(mapper.toState(first));
                ForecastDescriptor a = first.process(point, 0L);
                ForecastDescriptor b = shadow.process(point, 0L);
                assertEquals(a.getRCFScore(), b.getRCFScore(), 1e-6);
                first.process(point, 0L);
            }

            // serialize + deserialize
            RCFCasterMapper mapper = new RCFCasterMapper();
            RCFCaster second = mapper.toModel(mapper.toState(first));

            // update re-instantiated forest
            for (int i = 0; i < 100; i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
                ForecastDescriptor firstResult = first.process(point, 0L);
                ForecastDescriptor secondResult = second.process(point, 0L);
                assertEquals(firstResult.getDataConfidence(), secondResult.getDataConfidence(), 1e-10);
                verifyForecast(firstResult, secondResult, 1);
            }
        }
    }
}
