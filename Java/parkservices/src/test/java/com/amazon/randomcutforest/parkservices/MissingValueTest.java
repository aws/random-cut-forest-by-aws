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
import java.util.List;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;

public class MissingValueTest {
    @ParameterizedTest
    @EnumSource(ImputationMethod.class)
    public void testConfidence(ImputationMethod method) {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int baseDimensions = 1;

        long count = 0;

        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).imputationMethod(method)
                .fillValues(new double[] { 3 }).forestMode(ForestMode.STREAMING_IMPUTE)
                .transformMethod(TransformMethod.NORMALIZE).autoAdjust(true).build();

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
                double[] actual = new double[] { (rcfPoint[3] * scale) + shift };
                if (method == ImputationMethod.ZERO) {
                    assertEquals(0, actual[0], 0.001d);
                } else if (method == ImputationMethod.FIXED_VALUES) {
                    assertEquals(3.0d, actual[0], 0.001d);
                }
            } else {
                AnomalyDescriptor result = forest.process(point, newStamp);
                if ((count > 100 && count < 300) || count >= 326) {
                    // The first 65+ observations gives 0 confidence.
                    // Confidence start increasing after 1 observed point
                    assertTrue(result.getDataConfidence() > lastConfidence);
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
