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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;

public class IgnoreTest {
    @Test
    public void testAnomalies() {
        // Initialize the forest parameters
        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int baseDimensions = 1;

        long count = 0;
        int dimensions = baseDimensions * shingleSize;

        // Build the ThresholdedRandomCutForest
        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STREAMING_IMPUTE)
                .transformMethod(TransformMethod.NORMALIZE).autoAdjust(true)
                .ignoreNearExpectedFromAboveByRatio(new double[] { 0.1 })
                .ignoreNearExpectedFromBelowByRatio(new double[] { 0.1 }).build();

        // Generate the list of doubles
        List<Double> randomDoubles = generateUniformRandomDoubles();

        // List to store detected anomaly indices
        List<Integer> anomalies = new ArrayList<>();

        // Process each data point through the forest
        for (double val : randomDoubles) {
            double[] point = new double[] { val };
            long newStamp = 100 * count;

            AnomalyDescriptor result = forest.process(point, newStamp);

            if (result.getAnomalyGrade() != 0) {
                anomalies.add((int) count);
            }
            ++count;
        }

        // Expected anomalies
        List<Integer> expectedAnomalies = Arrays.asList(273, 283, 505, 1323);

        System.out.println("Anomalies detected at indices: " + anomalies);

        // Verify that all expected anomalies are detected
        assertTrue(anomalies.containsAll(expectedAnomalies),
                "Anomalies detected do not contain all expected anomalies");
    }

    public static List<Double> generateUniformRandomDoubles() {
        // Set fixed times for reproducibility
        LocalDateTime startTime = LocalDateTime.of(2020, 1, 1, 0, 0, 0);
        LocalDateTime endTime = LocalDateTime.of(2020, 1, 2, 0, 0, 0);
        long totalIntervals = ChronoUnit.MINUTES.between(startTime, endTime);

        // Generate timestamps (not used but kept for completeness)
        List<LocalDateTime> timestamps = new ArrayList<>();
        for (int i = 0; i < totalIntervals; i++) {
            timestamps.add(startTime.plusMinutes(i));
        }

        // Initialize variables
        Random random = new Random(0); // For reproducibility
        double level = 0;
        List<Double> logCounts = new ArrayList<>();

        // Decide random change points where level will change
        int numChanges = random.nextInt(6) + 5; // Random number between 5 and 10 inclusive

        Set<Integer> changeIndicesSet = new TreeSet<>();
        changeIndicesSet.add(0); // Ensure the first index is included

        while (changeIndicesSet.size() < numChanges) {
            int idx = random.nextInt((int) totalIntervals - 1) + 1; // Random index between 1 and totalIntervals -1
            changeIndicesSet.add(idx);
        }

        List<Integer> changeIndices = new ArrayList<>(changeIndicesSet);

        // Generate levels at each change point
        List<Double> levels = new ArrayList<>();
        for (int i = 0; i < changeIndices.size(); i++) {
            if (i == 0) {
                level = random.nextDouble() * 10; // Starting level between 0 and 10
            } else {
                double increment = -2 + random.nextDouble() * 7; // Random increment between -2 and 5
                level = Math.max(0, level + increment);
            }
            levels.add(level);
        }

        // Now generate logCounts for each timestamp with even smoother transitions
        int currentLevelIndex = 0;
        for (int idx = 0; idx < totalIntervals; idx++) {
            if (currentLevelIndex + 1 < changeIndices.size() && idx >= changeIndices.get(currentLevelIndex + 1)) {
                currentLevelIndex++;
            }
            level = levels.get(currentLevelIndex);
            double sineWave = Math.sin((idx % 300) * (Math.PI / 150)) * 0.05 * level;
            double noise = (-0.01 * level) + random.nextDouble() * (0.02 * level); // Noise between -0.01*level and
                                                                                   // 0.01*level
            double count = Math.max(0, level + sineWave + noise);
            logCounts.add(count);
        }

        // Introduce controlled changes for anomaly detection testing
        for (int changeIdx : changeIndices) {
            if (changeIdx + 10 < totalIntervals) {
                logCounts.set(changeIdx + 5, logCounts.get(changeIdx + 5) * 1.05); // 5% increase
                logCounts.set(changeIdx + 10, logCounts.get(changeIdx + 10) * 1.10); // 10% increase
            }
        }

        // Output the generated logCounts
        System.out.println("Generated logCounts of size: " + logCounts.size());
        return logCounts;
    }
}
