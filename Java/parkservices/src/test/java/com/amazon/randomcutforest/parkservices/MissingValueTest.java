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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;

public class MissingValueTest {
    private static class EnumAndValueProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) {
            return Stream.of(ImputationMethod.PREVIOUS, ImputationMethod.ZERO, ImputationMethod.FIXED_VALUES)
                    .flatMap(method -> Stream.of(4, 8, 16) // Example shingle sizes
                            .map(shingleSize -> Arguments.of(method, shingleSize)));
        }
    }

    @ParameterizedTest
    @ArgumentsSource(EnumAndValueProvider.class)
    public void testConfidence(ImputationMethod method, int shingleSize) {
        // Create and populate a random cut forest

        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int baseDimensions = 1;

        long count = 0;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest.Builder forestBuilder = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).imputationMethod(method)
                .forestMode(ForestMode.STREAMING_IMPUTE).transformMethod(TransformMethod.NORMALIZE).autoAdjust(true);

        if (method == ImputationMethod.FIXED_VALUES) {
            // we cannot pass fillValues when the method is not fixed values. Otherwise, we
            // will impute
            // filled in values irregardless of imputation method
            forestBuilder.fillValues(new double[] { 3 });
        }

        ThresholdedRandomCutForest forest = forestBuilder.build();

        // Define the size and range
        int size = 400;
        double min = 200.0;
        double max = 240.0;

        // Generate the list of doubles
        List<Double> randomDoubles = generateUniformRandomDoubles(size, min, max);

        double lastConfidence = 0;
        for (double val : randomDoubles) {
            double[] point = new double[] { val };
            long newStamp = 100 * count;
            if (count >= 300 && count < 325) {
                // drop observations
                AnomalyDescriptor result = forest.process(new double[] { Double.NaN }, newStamp,
                        generateIntArray(point.length));
                if (count > 300) {
                    // confidence start decreasing after 1 missing point
                    assertTrue(result.getDataConfidence() < lastConfidence, "count " + count);
                }
                lastConfidence = result.getDataConfidence();
                float[] rcfPoint = result.getRCFPoint();
                double scale = result.getScale()[0];
                double shift = result.getShift()[0];
                double[] actual = new double[] { (rcfPoint[shingleSize - 1] * scale) + shift };
                if (method == ImputationMethod.ZERO) {
                    assertEquals(0, actual[0], 0.001d);
                    if (count == 300) {
                        assertTrue(result.getAnomalyGrade() > 0);
                    }
                } else if (method == ImputationMethod.FIXED_VALUES) {
                    assertEquals(3.0d, actual[0], 0.001d);
                    if (count == 300) {
                        assertTrue(result.getAnomalyGrade() > 0);
                    }
                } else if (method == ImputationMethod.PREVIOUS) {
                    assertEquals(0, result.getAnomalyGrade(), 0.001d,
                            "count: " + count + " actual: " + Arrays.toString(actual));
                }
            } else {
                AnomalyDescriptor result = forest.process(point, newStamp);
                // after 325, we have a period of confidence decreasing. After that, confidence
                // starts increasing again.
                // We are not sure where the confidence will start increasing after decreasing.
                // So we start check the behavior after 325 + shingleSize.
                int backupPoint = 325 + shingleSize;
                if ((count > 100 && count < 300) || count >= backupPoint) {
                    // The first 65+ observations gives 0 confidence.
                    // Confidence start increasing after 1 observed point
                    assertTrue(result.getDataConfidence() > lastConfidence,
                            String.format(Locale.ROOT, "count: %d, confidence: %f, last confidence: %f", count,
                                    result.getDataConfidence(), lastConfidence));
                } else if (count < 325 && count > 300) {
                    assertTrue(result.getDataConfidence() < lastConfidence,
                            String.format(Locale.ROOT, "count: %d, confidence: %f, last confidence: %f", count,
                                    result.getDataConfidence(), lastConfidence));
                }
                lastConfidence = result.getDataConfidence();
            }
            ++count;
        }
    }

    public static int[] generateIntArray(int size) {
        int[] intArray = new int[size];
        for (int i = 0; i < size; i++) {
            intArray[i] = i;
        }
        return intArray;
    }

    public static List<Double> generateUniformRandomDoubles(int size, double min, double max) {
        List<Double> randomDoubles = new ArrayList<>(size);
        Random random = new Random(0);

        for (int i = 0; i < size; i++) {
            double randomValue = min + (max - min) * random.nextDouble();
            randomDoubles.add(randomValue);
        }

        return randomDoubles;
    }
}
