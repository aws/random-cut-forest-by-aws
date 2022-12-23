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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;
import com.amazon.randomcutforest.parkservices.RCFCaster;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

public class RCFCasterMapperTest {
    @ParameterizedTest
    @ValueSource(ints = { 1, 2 })
    public void testRoundTripStandardShingleSizeEight(int inputLength) {
        int shingleSize = 8;
        int dimensions = inputLength * shingleSize;
        int forecastHorizon = shingleSize * 3;
        for (int trials = 0; trials < 1; trials++) {

            long seed = new Random().nextLong();

            // note shingleSize == 8
            RCFCaster first = RCFCaster.builder().compact(true).dimensions(dimensions).precision(Precision.FLOAT_32)
                    .randomSeed(seed).internalShinglingEnabled(true).anomalyRate(0.01).shingleSize(shingleSize)
                    .calibration(Calibration.MINIMAL).forecastHorizon(forecastHorizon)
                    .transformMethod(TransformMethod.NORMALIZE).build();

            Random r = new Random();
            for (int i = 0; i < 2000 + new Random().nextInt(1000); i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
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

    void verifyForecast(ForecastDescriptor firstResult, ForecastDescriptor secondResult, int inputLength) {
        RangeVector firstForecast = firstResult.getTimedForecast().rangeVector;
        RangeVector secondForecast = secondResult.getTimedForecast().rangeVector;

        float[] firstErrorP50 = firstResult.getObservedErrorDistribution().values;
        float[] secondErrorP50 = secondResult.getObservedErrorDistribution().values;

        float[] firstUpperError = firstResult.getObservedErrorDistribution().upper;
        float[] secondUpperError = secondResult.getObservedErrorDistribution().upper;

        float[] firstLowerError = firstResult.getObservedErrorDistribution().lower;
        float[] secondLowerError = secondResult.getObservedErrorDistribution().lower;

        DiVector firstRmse = firstResult.getErrorRMSE();
        DiVector secondRmse = secondResult.getErrorRMSE();

        float[] firstMean = firstResult.getErrorMean();
        float[] secondMean = secondResult.getErrorMean();

        float[] firstCalibration = firstResult.getCalibration();
        float[] secondCalibration = secondResult.getCalibration();

        // block corresponding to the past; print the errors
        for (int i = firstForecast.values.length / inputLength - 1; i >= 0; i--) {
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                assertEquals(firstMean[k], secondMean[k], 1e-10);
                assertEquals(firstRmse.high[k], secondRmse.high[k], 1e-10);
                assertEquals(firstRmse.low[k], secondRmse.low[k], 1e-10);
                assertEquals(firstErrorP50[k], secondErrorP50[k], 1e-10);
                assertEquals(firstUpperError[k], secondUpperError[k], 1e-10);
                assertEquals(firstLowerError[k], secondLowerError[k], 1e-10);
                assertEquals(firstCalibration[k], secondCalibration[k], 1e-10);
            }
        }

        // block corresponding to the future; the projections and the projected errors
        for (int i = 0; i < firstForecast.values.length / inputLength; i++) {
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                assertEquals(firstForecast.values[k], secondForecast.values[k], 1e-10);
                assertEquals(firstForecast.upper[k], secondForecast.upper[k], 1e-10);
                assertEquals(firstForecast.lower[k], secondForecast.lower[k], 1e-10);
            }
        }
    }

    @Test
    public void testNotFullyInitialized() {
        int inputLength = 1;
        int shingleSize = 8;
        int dimensions = inputLength * shingleSize;
        int forecastHorizon = shingleSize * 3;
        int outputAfter = 32;
        for (int trials = 0; trials < 1; trials++) {

            long seed = new Random().nextLong();

            // note shingleSize == 8
            RCFCaster first = RCFCaster.builder().compact(true).dimensions(dimensions).precision(Precision.FLOAT_32)
                    .randomSeed(seed).internalShinglingEnabled(true).anomalyRate(0.01).shingleSize(shingleSize)
                    .calibration(Calibration.MINIMAL).forecastHorizon(forecastHorizon)
                    .transformMethod(TransformMethod.NORMALIZE).outputAfter(outputAfter).build();

            Random r = new Random();
            for (int i = 0; i < new Random().nextInt(outputAfter); i++) {
                double[] point = r.ints(inputLength, 0, 50).asDoubleStream().toArray();
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
