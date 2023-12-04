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

import static com.amazon.randomcutforest.config.ImputationMethod.RCF;
import static com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys.generateShingledData;
import static java.lang.Math.min;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

@Tag("functional")
public class ForecastTest {

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void basicAndIdempotence(TransformMethod method) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = 1L;

        int length = 4 * sampleSize;
        int outputAfter = 128;

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).outputAfter(outputAfter).transformMethod(method).build();

        // as the ratio of amplitude (signal) to noise is changed, the estimation range
        // in forecast
        // (or any other inference) should increase
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 10, seed,
                baseDimensions);

        System.out.println(dataWithKeys.changes.length + " anomalies injected ");
        double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, false);

        assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

        int horizon = 20;
        if (method == TransformMethod.NORMALIZE_DIFFERENCE || method == TransformMethod.DIFFERENCE) {
            horizon = min(horizon, shingleSize / 2 + 1);
        }
        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            // forecast first; change centrality to achieve a control over the sampling
            // setting centrality = 0 would correspond to random sampling from the leaves
            // reached by
            // impute visitor

            TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
            RangeVector forecast = extrapolate.rangeVector;
            assert (forecast.values.length == horizon);
            assert (extrapolate.timeStamps.length == horizon);
            assert (extrapolate.lowerTimeStamps.length == horizon);
            assert (extrapolate.upperTimeStamps.length == horizon);

            RangeVector alternative = forest.extrapolate(horizon, true, 1.0).rangeVector;

            // repeated invocations of extrapolate should return same result
            // for the same values of correction,centrality
            assertArrayEquals(forecast.values, alternative.values, 1e-6f);
            assertArrayEquals(forecast.lower, alternative.lower, 1e-6f);
            assertArrayEquals(forecast.upper, alternative.upper, 1e-6f);

            for (int i = 0; i < horizon; i++) {
                // check ranges
                if (j > sampleSize) {
                    assert (extrapolate.timeStamps[i] == j + i);
                    assert (extrapolate.upperTimeStamps[i] == j + i);
                    assert (extrapolate.lowerTimeStamps[i] == j + i);
                }
                assert (forecast.values[i] >= forecast.lower[i]);
                assert (forecast.values[i] <= forecast.upper[i]);
                // compute errors
                if (j > outputAfter + shingleSize - 1 && j + i < dataWithKeys.data.length) {
                    double t = dataWithKeys.data[j + i][0] - forecast.values[i];
                    error[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.lower[i];
                    lowerError[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.upper[i];
                    upperError[i] += t * t;
                }
            }
            forest.process(dataWithKeys.data[j], j);
        }

        System.out.println(forest.getTransformMethod().name() + " RMSE (as horizon increases) ");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

    @ParameterizedTest
    @CsvSource({ "NORMALIZE,true", "NORMALIZE,false", "SUBTRACT_MA,true", "SUBTRACT_MA,false", "WEIGHTED,true",
            "WEIGHTED,false" })
    public void linearShift(String methodString, String normalizeTime) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = 0L;

        int length = 10 * sampleSize;
        int outputAfter = 128;

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).timeDecay(1.0 / 1024).outputAfter(outputAfter)
                .transformMethod(TransformMethod.valueOf(methodString))
                .normalizeTime(Boolean.parseBoolean(normalizeTime)).build();

        // as the ratio of amplitude (signal) to noise is changed, the estimation range
        // in forecast
        // (or any other inference) should increase
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 10, seed,
                baseDimensions, true);

        System.out.println(dataWithKeys.changes.length + " anomalies injected ");
        double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, false);

        assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

        // the following constraint is for differencing based methods
        int horizon = shingleSize / 2 + 1;

        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            // forecast first; change centrality to achieve a control over the sampling
            // setting centrality = 0 would correspond to random sampling from the leaves
            // reached by
            // impute visitor
            TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
            RangeVector forecast = extrapolate.rangeVector;
            assert (forecast.values.length == horizon);
            assert (extrapolate.timeStamps.length == horizon);
            assert (extrapolate.lowerTimeStamps.length == horizon);
            assert (extrapolate.upperTimeStamps.length == horizon);

            RangeVector alternative = forest.extrapolate(horizon, true, 1.0).rangeVector;

            // repeated invocations of extrapolate should return same result
            // for the same values of correction,centrality
            assertArrayEquals(forecast.values, alternative.values, 1e-6f);
            assertArrayEquals(forecast.lower, alternative.lower, 1e-6f);
            assertArrayEquals(forecast.upper, alternative.upper, 1e-6f);

            for (int i = 0; i < horizon; i++) {

                if (j > outputAfter) {
                    assert (extrapolate.timeStamps[i] == i + j);
                    assert (extrapolate.upperTimeStamps[i] == i + j);
                    assert (extrapolate.lowerTimeStamps[i] == i + j);
                }
                // check ranges
                assert (forecast.values[i] >= forecast.lower[i]);
                assert (forecast.values[i] <= forecast.upper[i]);
                // compute errors
                if (j > outputAfter + shingleSize - 1 && j + i < dataWithKeys.data.length) {
                    double t = dataWithKeys.data[j + i][0] - forecast.values[i];
                    error[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.lower[i];
                    lowerError[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.upper[i];
                    upperError[i] += t * t;
                }
            }
            forest.process(dataWithKeys.data[j], j);
        }

        System.out.println(forest.getTransformMethod().name() + " RMSE (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

    @ParameterizedTest
    @CsvSource({ "DIFFERENCE,true", "DIFFERENCE,false", "NORMALIZE_DIFFERENCE,true", "NORMALIZE_DIFFERENCE,false" })
    public void linearShiftDifference(String methodString, String normalizeTime) {
        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        // use same seed as previous test
        long seed = 0L;

        int length = 10 * sampleSize;
        int outputAfter = 128;

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                .shingleSize(shingleSize).timeDecay(1.0 / 1024).outputAfter(outputAfter)
                .transformMethod(TransformMethod.valueOf(methodString))
                .normalizeTime(Boolean.parseBoolean(normalizeTime)).build();

        // as the ratio of amplitude (signal) to noise is changed, the estimation range
        // in forecast
        // (or any other inference) should increase
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 10, seed,
                baseDimensions, true);

        System.out.println(dataWithKeys.changes.length + " anomalies injected ");
        double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, false);

        assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

        // the following constraint is for differencing based methods
        // the differenced values will be noisy in the presence of anomalies
        // the example demonstrates that the best forecaster need not be the best
        // anomaly detector, even from a restricted family of algorithms
        int horizon = shingleSize / 2 + 1;

        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        for (int j = 0; j < dataWithKeys.data.length; j++) {

            TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
            RangeVector forecast = extrapolate.rangeVector;
            assert (forecast.values.length == horizon);
            assert (extrapolate.timeStamps.length == horizon);
            assert (extrapolate.lowerTimeStamps.length == horizon);
            assert (extrapolate.upperTimeStamps.length == horizon);

            RangeVector alternative = forest.extrapolate(horizon, true, 1.0).rangeVector;

            // repeated invocations of extrapolate should return same result
            // for the same values of correction,centrality
            assertArrayEquals(forecast.values, alternative.values, 1e-6f);
            assertArrayEquals(forecast.lower, alternative.lower, 1e-6f);
            assertArrayEquals(forecast.upper, alternative.upper, 1e-6f);

            for (int i = 0; i < horizon; i++) {
                // check ranges
                assertEquals(extrapolate.timeStamps[i], 0);
                assertEquals(extrapolate.upperTimeStamps[i], 0);
                assertEquals(extrapolate.lowerTimeStamps[i], 0);
                assert (forecast.values[i] >= forecast.lower[i]);
                assert (forecast.values[i] <= forecast.upper[i]);
                // compute errors
                if (j > outputAfter + shingleSize - 1 && j + i < dataWithKeys.data.length) {
                    double t = dataWithKeys.data[j + i][0] - forecast.values[i];
                    error[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.lower[i];
                    lowerError[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.upper[i];
                    upperError[i] += t * t;
                }
            }
            forest.process(dataWithKeys.data[j], 0L);
        }

        System.out.println(forest.getTransformMethod().name() + " RMSE (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    void timeAugmentedTest(boolean normalize) {
        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;

        int baseDimensions = 1;
        int horizon = 10;

        int count = 0;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).internalShinglingEnabled(true).precision(precision).anomalyRate(0.01)
                .forestMode(ForestMode.TIME_AUGMENTED).normalizeTime(normalize).build();

        long seed = new Random().nextLong();
        double[] data = new double[] { 1.0 };
        System.out.println("seed = " + seed);
        Random rng = new Random(seed);

        for (int i = 0; i < 200; i++) {
            long time = 1000L * count + rng.nextInt(100);
            forest.process(data, time);
            ++count;
        }
        TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
        RangeVector range = extrapolate.rangeVector;
        assert (range.values.length == baseDimensions * horizon);
        assert (extrapolate.timeStamps.length == horizon);
        assert (extrapolate.lowerTimeStamps.length == horizon);
        assert (extrapolate.upperTimeStamps.length == horizon);

        /*
         * the forecasted time stamps should be close to 1000 * (count + i) the data
         * values should remain as in data[]
         */

        for (int i = 0; i < horizon; i++) {
            assertEquals(range.values[i], data[0]);
            assertEquals(range.upper[i], data[0]);
            assertEquals(range.lower[i], data[0]);
            assert (Math.abs(Math.round(extrapolate.timeStamps[i] * 0.001) - count - i) <= 1);
            assert (extrapolate.timeStamps[i] >= extrapolate.lowerTimeStamps[i]);
            assert (extrapolate.upperTimeStamps[i] >= extrapolate.timeStamps[i]);
        }
    }

    @ParameterizedTest
    @EnumSource(TransformMethod.class)
    public void streamingImputeTest(TransformMethod method) {
        int shingleSize = 8;
        int numberOfTrees = 100;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;
        int outputAfter = sampleSize;

        // change this to try different number of attributes,
        int baseDimensions = 1;

        int dropped = 0;
        long seed = 2022L;

        // the following simulates random drops
        long dropSeed = 7L;
        Random dropPRG = new Random(dropSeed);

        System.out.println("seed = " + seed);
        System.out.println("dropping seed = " + dropSeed);

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STREAMING_IMPUTE)
                .transformMethod(method).imputationMethod(RCF).build();

        // limited to shingleSize/2+1 due to the differenced methods
        int horizon = shingleSize / 2 + 1;

        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                100, 5, seed, baseDimensions, true);
        System.out.println(dataWithKeys.changes.length + " anomalies injected ");
        for (int j = 0; j < dataWithKeys.data.length; j++) {

            if (dropPRG.nextDouble() < 0.2) {
                ++dropped;
            } else {

                // note that the forecast does not change without a new reading in streaming
                // impute
                // in this case the forecast corresponds to j+1 .. j + horizon
                // so we will add the j'th entry and then measure error against j+1 ...
                // j+horizon values

                long newStamp = 1000L * j + 10 * dropPRG.nextInt(10) - 5;
                forest.process(dataWithKeys.data[j], newStamp);
                TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
                RangeVector forecast = extrapolate.rangeVector;
                assert (forecast.values.length == horizon);
                assert (extrapolate.timeStamps.length == horizon);

                RangeVector alternative = forest.extrapolate(horizon, true, 1.0).rangeVector;

                // repeated invocations of extrapolate should return same result
                // for the same values of correction,centrality
                assertArrayEquals(forecast.values, alternative.values, 1e-6f);
                assertArrayEquals(forecast.lower, alternative.lower, 1e-6f);
                assertArrayEquals(forecast.upper, alternative.upper, 1e-6f);

                for (int i = 0; i < horizon; i++) {
                    // check ranges
                    assert (forecast.values[i] >= forecast.lower[i]);
                    assert (forecast.values[i] <= forecast.upper[i]);
                    assertEquals(extrapolate.timeStamps[i], 0);
                    assertEquals(extrapolate.upperTimeStamps[i], 0);
                    assertEquals(extrapolate.lowerTimeStamps[i], 0);
                    // compute errors
                    // NOTE the +1 since we are predicting the unseen values in the data
                    if (j > outputAfter + shingleSize - 1 && j + i + 1 < dataWithKeys.data.length) {
                        double t = dataWithKeys.data[j + i + 1][0] - forecast.values[i];
                        error[i] += t * t;
                        t = dataWithKeys.data[j + i + 1][0] - forecast.lower[i];
                        lowerError[i] += t * t;
                        t = dataWithKeys.data[j + i + 1][0] - forecast.upper[i];
                        upperError[i] += t * t;
                    }
                }
            }
        }
        System.out.println("Impute with " + dropped + " dropped values from " + dataWithKeys.data.length + " values");
        System.out.println(forest.getTransformMethod().name() + " RMSE (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i - dropped);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i - dropped);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i - dropped);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

}
