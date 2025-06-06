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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class TransformTest {

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void AnomalyTest(TransformMethod method) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 10;
        int length = 40 * sampleSize;
        int totalcount = 0;
        for (int i = 0; i < numTrials; i++) {
            int numberOfTrees = 30 + rng.nextInt(20);
            int outputAfter = 32 + rng.nextInt(50);
            // shingleSize 1 is not recommended for complicated input
            int shingleSize = 2 + rng.nextInt(15);
            int baseDimensions = 1 + rng.nextInt(5);
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                    .numberOfTrees(numberOfTrees).randomSeed(forestSeed).outputAfter(outputAfter).alertOnce(true)
                    .transformMethod(method).internalShinglingEnabled(true).shingleSize(shingleSize).build();

            int count = 0;
            double[] point = new double[baseDimensions];
            double[] anomalyPoint = new double[baseDimensions];
            for (int j = 0; j < baseDimensions; j++) {
                point[j] = 50 - rng.nextInt(100);
                int sign = (rng.nextDouble() < 0.5) ? -1 : 1;
                anomalyPoint[j] = point[j] + sign * (10 - rng.nextInt(5));
            }
            int anomalyAt = outputAfter + rng.nextInt(length / 2);
            for (int j = 0; j < anomalyAt; j++) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    ++count;
                }
            }
            assertEquals(0, count);
            assertTrue(first.process(anomalyPoint, 0L).getAnomalyGrade() > 0);
            for (int j = anomalyAt + 1; j < length; j++) {
                AnomalyDescriptor firstResult = first.process(point, 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    ++count;
                }
            }
            // differencing introduces cascades
            totalcount += count;
        }
        System.out.println(totalcount);
        int finalTotalcount = totalcount;
        assertTrue(
                totalcount < numTrials || method == TransformMethod.DIFFERENCE
                        || method == TransformMethod.NORMALIZE_DIFFERENCE,
                () -> String.format(
                        "Assertion failed: totalcount=%d, numTrials=%d, method=%s, sampleSize=%d, length=%d, seed=%d",
                        finalTotalcount, numTrials, method, sampleSize, length, seed));
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class, names = { "NONE", "NORMALIZE" })
    public void AnomalyTestSine1D(TransformMethod method) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 50;
        int length = 4 * sampleSize;
        int found = 0;
        int count = 0;
        double grade = 0;

        for (int i = 0; i < numTrials; i++) {
            int numberOfTrees = 50 + rng.nextInt(20);
            int outputAfter = 64 + rng.nextInt(50);
            int shingleSize = 8;
            int baseDimensions = 1; // multiple dimensions would have anti-correlations induced by
                                    // differring periods
            int dimensions = baseDimensions * shingleSize;
            long forestSeed = rng.nextLong();
            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().dimensions(dimensions)
                    .numberOfTrees(numberOfTrees).randomSeed(forestSeed).outputAfter(outputAfter)
                    .transformMethod(method).internalShinglingEnabled(true).shingleSize(shingleSize).build();
            double[][] data = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 0, rng.nextLong(),
                    baseDimensions, 0, false).data;

            int anomalyAt = outputAfter + rng.nextInt(length / 2);
            for (int j = 0; j < baseDimensions; j++) {
                int sign = (rng.nextDouble() < 0.5) ? -1 : 1;
                // large obvious spike
                data[anomalyAt][j] += sign * 100;
            }

            for (int j = 0; j < length; j++) {
                AnomalyDescriptor firstResult = first.process(data[j], 0L);
                if (firstResult.getAnomalyGrade() > 0) {
                    // detection can be late
                    if (j + firstResult.getRelativeIndex() == anomalyAt) {
                        ++found;
                    }
                    ++count;
                    grade += firstResult.getAnomalyGrade();
                }

            }
        }
        System.out.println(found);
        // catch anomalies 80% of the time
        assertTrue(found > 0.8 * numTrials);

        // precision is not terrible
        assertTrue(count < 2 * numTrials);

        // average grade is closer to found
        assertTrue(grade < 1.5 * numTrials);
    }

    @ParameterizedTest
    @EnumSource(value = TransformMethod.class, names = { "NORMALIZE", "NORMALIZE_DIFFERENCE", "DIFFERENCE" })
    public void RCFCastTest(TransformMethod method) {
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        Random rng = new Random(seed);
        int numTrials = 1;
        int length = sampleSize / 2;
        int forecastHorizon = 2;
        for (int i = 0; i < numTrials; i++) {
            int numberOfTrees = 30 + rng.nextInt(20);
            int outputAfter = 32 + rng.nextInt(50);
            // shingleSize 1 is not recommended for complicated input
            int shingleSize = 4 + rng.nextInt(5);
            int baseDimensions = 1;
            int offset = rng.nextInt(10);
            int dimensions = baseDimensions * shingleSize;
            RCFCaster first = new RCFCaster.Builder().dimensions(dimensions).numberOfTrees(numberOfTrees).randomSeed(0)
                    .outputAfter(outputAfter).alertOnce(true).forecastHorizon(forecastHorizon).transformMethod(method)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).build();

            for (int j = 0; j < length; j++) {
                ForecastDescriptor firstResult = first.process(new double[] { j + offset }, 0L);
                if (j >= outputAfter - 1) {
                    for (int y = 0; y < forecastHorizon; y++) {
                        assertTrue(Math.abs(
                                firstResult.getTimedForecast().rangeVector.values[y] - (j + offset + 1 + y)) < 0.3);
                    }
                }
            }
        }
    }

}
