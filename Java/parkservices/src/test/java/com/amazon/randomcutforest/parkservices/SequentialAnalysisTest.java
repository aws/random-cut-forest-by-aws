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

import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static java.lang.Math.min;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.parkservices.returntypes.AnalysisDescriptor;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class SequentialAnalysisTest {

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void AnomalyTest(TransformMethod method) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 1; // just once since testing exact equality
        int length = 40 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int numberOfTrees = 30 + rng.nextInt(20);
            int outputAfter = 1 + rng.nextInt(50);
            int shingleSize = 1 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            double timeDecay = 0.1 / sampleSize;
            double transformDecay = 1.0 / sampleSize;
            double fraction = 1.0 * outputAfter / sampleSize;
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).numberOfTrees(numberOfTrees).randomSeed(forestSeed).outputAfter(outputAfter)
                    .transformMethod(method).timeDecay(timeDecay).transformDecay(transformDecay)
                    .internalShinglingEnabled(true).initialAcceptFraction(fraction).shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            List<AnomalyDescriptor> result = SequentialAnalysis.detectAnomalies(dataWithKeys.data, shingleSize,
                    sampleSize, numberOfTrees, timeDecay, outputAfter, method, transformDecay, forestSeed);

            int count = 0;
            for (double[] point : dataWithKeys.data) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    assertEquals(firstResult.getAnomalyGrade(), result.get(count).getAnomalyGrade(), 1e-3);
                    assertEquals(firstResult.getInternalTimeStamp(), result.get(count).getInternalTimeStamp());
                    ++count;
                }
            }
            assertTrue(count == result.size());
        }
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void AnomalyTest2(TransformMethod method) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 1; // just once since testing exact equality
        int length = 40 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int outputAfter = sampleSize / 4;
            int shingleSize = 1 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            double timeDecay = 0.1 / sampleSize;
            double fraction = 1.0 * outputAfter / sampleSize;
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).randomSeed(forestSeed).transformMethod(method).timeDecay(timeDecay)
                    .internalShinglingEnabled(true).transformDecay(timeDecay).initialAcceptFraction(fraction)
                    .shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            List<AnomalyDescriptor> result = SequentialAnalysis.detectAnomalies(dataWithKeys.data, shingleSize,
                    sampleSize, timeDecay, method, forestSeed);

            int count = 0;
            for (double[] point : dataWithKeys.data) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    assertEquals(firstResult.getAnomalyGrade(), result.get(count).getAnomalyGrade(), 1e-3);
                    assertEquals(firstResult.getInternalTimeStamp(), result.get(count).getInternalTimeStamp());
                    assertEquals(firstResult.getRCFScore(), result.get(count).getRCFScore(), 1e-3);
                    ++count;
                }
            }
            assertTrue(count == result.size());
        }
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void AnomalyTest3(TransformMethod method) {
        int sampleSize = DEFAULT_SAMPLE_SIZE;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 1; // just once since testing exact equality
        int length = 40 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int outputAfter = sampleSize / 4;
            int shingleSize = 1 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            double timeDecay = 0.1 / sampleSize;
            double transformDecay = (1.0 + rng.nextDouble()) / sampleSize;
            double fraction = 1.0 * outputAfter / sampleSize;
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).randomSeed(forestSeed).transformMethod(method).timeDecay(timeDecay)
                    .internalShinglingEnabled(true).transformDecay(transformDecay).initialAcceptFraction(fraction)
                    .shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            List<AnomalyDescriptor> result = SequentialAnalysis.detectAnomalies(dataWithKeys.data, shingleSize,
                    timeDecay, method, transformDecay, forestSeed);

            int count = 0;
            for (double[] point : dataWithKeys.data) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    assertEquals(firstResult.getAnomalyGrade(), result.get(count).getAnomalyGrade(), 1e-3);
                    assertEquals(firstResult.getInternalTimeStamp(), result.get(count).getInternalTimeStamp());
                    assertEquals(firstResult.getRCFScore(), result.get(count).getRCFScore(), 1e-3);
                    ++count;
                }
            }
            assertTrue(count == result.size());
        }
    }

    @ParameterizedTest
    @EnumSource(Calibration.class)
    public void ForecasterTest(Calibration calibration) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 1; // just once since testing exact equality
        int length = 4 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            int numberOfTrees = 50;
            int outputAfter = 1 + rng.nextInt(50);
            int shingleSize = 2 + rng.nextInt(15);
            int forecastHorizon = min(4 * shingleSize, 10);
            int errorHorizon = 100;
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            double timeDecay = 0.1 / sampleSize;
            double transformDecay = 1.0 / sampleSize;
            double fraction = 1.0 * outputAfter / sampleSize;
            RCFCaster first = new RCFCaster.Builder().dimensions(dimensions).numberOfTrees(numberOfTrees)
                    .randomSeed(forestSeed).outputAfter(outputAfter).transformMethod(TransformMethod.NORMALIZE)
                    .timeDecay(timeDecay).transformDecay(transformDecay).internalShinglingEnabled(true)
                    .forecastHorizon(forecastHorizon).errorHorizon(errorHorizon).calibration(calibration)
                    .initialAcceptFraction(fraction).shingleSize(shingleSize).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    rng.nextLong(), baseDimensions);

            AnalysisDescriptor descriptor = SequentialAnalysis.forecastWithAnomalies(dataWithKeys.data, shingleSize,
                    sampleSize, timeDecay, outputAfter, TransformMethod.NORMALIZE, transformDecay, forecastHorizon,
                    errorHorizon, 0.1, calibration, forestSeed);

            List<AnomalyDescriptor> result = descriptor.getAnomalies();

            int count = 0;
            ForecastDescriptor last = null;
            for (double[] point : dataWithKeys.data) {
                ForecastDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    assertEquals(firstResult.getAnomalyGrade(), result.get(count).getAnomalyGrade(), 1e-3);
                    assertEquals(firstResult.getInternalTimeStamp(), result.get(count).getInternalTimeStamp());
                    assertEquals(firstResult.getRCFScore(), result.get(count).getRCFScore(), 1e-3);
                    ++count;
                }
                last = firstResult;
            }
            assertTrue(count == result.size());
            RangeVector sequential = descriptor.getForecastDescriptor().getTimedForecast().rangeVector;
            RangeVector current = last.getTimedForecast().rangeVector;
            assertArrayEquals(current.values, sequential.values, 1e-3f);
            assertArrayEquals(current.upper, sequential.upper, 1e-3f);
            assertArrayEquals(current.lower, sequential.lower, 1e-3f);
            assertArrayEquals(descriptor.getForecastDescriptor().getCalibration(), last.getCalibration(), 1e-3f);
            assertArrayEquals(descriptor.getForecastDescriptor().getErrorMean(), last.getErrorMean(), 1e-3f);
        }
    }

}
