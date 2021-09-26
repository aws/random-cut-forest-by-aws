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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

@Tag("functional")
public class ConsistencyTest {

    @Test
    public void InternalShinglingTest() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        int numTrials = 10;
        int length = 400 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).internalShinglingEnabled(true).shingleSize(shingleSize)
                    .randomSeed(seed).build();

            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    seed + i, baseDimensions);

            for (double[] point : dataWithKeys.data) {

                AnomalyDescriptor firstResult = first.process(point, 0L);
                assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-10);
                forest.update(point);
            }
        }
    }

    @Test
    public void ExternalShinglingTest() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        int numTrials = 10;
        int length = 400 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).internalShinglingEnabled(false).shingleSize(shingleSize)
                    .randomSeed(seed).build();

            RandomCutForest copyForest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).internalShinglingEnabled(false).shingleSize(1).randomSeed(seed)
                    .build();

            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build();

            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).shingleSize(1).anomalyRate(0.01).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(length, 50,
                    shingleSize, baseDimensions, seed);

            int gradeDifference = 0;

            for (double[] point : dataWithKeys.data) {

                AnomalyDescriptor firstResult = first.process(point, 0L);
                AnomalyDescriptor secondResult = second.process(point, 0L);

                assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-10);
                assertEquals(firstResult.getRcfScore(), copyForest.getAnomalyScore(point), 1e-10);
                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);

                if ((firstResult.getAnomalyGrade() > 0) != (secondResult.getAnomalyGrade() > 0)) {
                    ++gradeDifference;
                    // thresholded random cut forest uses shingle size in the corrector step
                    // this is supposed to be different
                }
                forest.update(point);
                copyForest.update(point);
            }
            assertTrue(gradeDifference > 0);
        }
    }

    @Test
    public void MixedShinglingTest() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();

        int numTrials = 10;
        int length = 400 * sampleSize;
        for (int i = 0; i < numTrials; i++) {

            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();

            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    seed + i, baseDimensions);

            double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions);

            assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

            int count = shingleSize - 1;
            // insert initial points
            for (int j = 0; j < shingleSize - 1; j++) {
                first.process(dataWithKeys.data[j], 0L);
            }

            int scoreDiff = 0;
            int gradeDiff = 0;
            int anomalyDiff = 0;

            for (int j = 0; j < shingledData.length; j++) {
                // validate eaulity of points
                for (int y = 0; y < baseDimensions; y++) {
                    assertEquals(dataWithKeys.data[count][y], shingledData[j][(shingleSize - 1) * baseDimensions + y],
                            1e-10);
                }

                AnomalyDescriptor firstResult = first.process(dataWithKeys.data[count], 0L);
                ++count;
                AnomalyDescriptor secondResult = second.process(shingledData[j], 0L);
                // the internal external would not have exact floating point inequality (because
                // the
                // order of arithmetic, etc. are different, because the tree cuts will
                // eventually be different) but the numbers will be
                // would be close -- however much of the computation depends on floating point
                // precision and the results can be slightly different
                if (Math.abs(firstResult.getRcfScore() - secondResult.getRcfScore()) > 0.005) {
                    ++scoreDiff;
                }
                if (firstResult.getAnomalyGrade() > 0 != secondResult.getAnomalyGrade() > 0) {
                    ++anomalyDiff;
                }
                if (Math.abs(firstResult.getAnomalyGrade() - secondResult.getAnomalyGrade()) > 0.005) {
                    ++gradeDiff;
                }
            }
            assertTrue(anomalyDiff < 2); // extremely unlikely; but can happen in a blue moon
            assertTrue(gradeDiff < length * 0.0001); // unlikely, but does happen
            assertTrue(scoreDiff < length * 0.001);
        }
    }

    double[][] generateShingledData(double[][] data, int shingleSize, int baseDimension) {
        int size = data.length - shingleSize + 1;
        double[][] answer = new double[size][];
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[][] history = new double[shingleSize][];
        int count = 0;

        for (int j = 0; j < size + shingleSize - 1; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shingleSize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                answer[count++] = getShinglePoint(history, entryIndex, shingleSize, baseDimension);
            }
        }
        return answer;
    }

    private static double[] getShinglePoint(double[][] recentPointsSeen, int indexOfOldestPoint, int shingleLength,
            int baseDimension) {
        double[] shingledPoint = new double[shingleLength * baseDimension];
        int count = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double[] point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            for (int i = 0; i < baseDimension; i++) {
                shingledPoint[count++] = point[i];
            }
        }
        return shingledPoint;
    }
}
