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

import static com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys.generateShingledData;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;
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

        int numTrials = 1; // just once since testing exact equality
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

        int numTrials = 1; // just once since testing exact equality
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

        int numTrials = 1; // test is exact equality, reducing the number of trials
        int length = 2000 * sampleSize;
        int testLength = length;
        for (int i = 0; i < numTrials; i++) {

            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).anomalyRate(0.01).build();

            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).shingleSize(shingleSize).anomalyRate(0.01).build();

            ThresholdedRandomCutForest third = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(dimensions).precision(Precision.FLOAT_32).randomSeed(seed)
                    .internalShinglingEnabled(false).shingleSize(1).anomalyRate(0.01).build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length + testLength, 50,
                    100, 5, seed + i, baseDimensions);

            double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, false);

            assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

            int count = shingleSize - 1;
            // insert initial points
            for (int j = 0; j < shingleSize - 1; j++) {
                first.process(dataWithKeys.data[j], 0L);
            }

            for (int j = 0; j < length; j++) {
                // validate eaulity of points
                for (int y = 0; y < baseDimensions; y++) {
                    assertEquals(dataWithKeys.data[count][y], shingledData[j][(shingleSize - 1) * baseDimensions + y],
                            1e-10);
                }

                AnomalyDescriptor firstResult = first.process(dataWithKeys.data[count], 0L);
                ++count;
                AnomalyDescriptor secondResult = second.process(shingledData[j], 0L);
                AnomalyDescriptor thirdResult = third.process(shingledData[j], 0L);

                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getAnomalyGrade(), secondResult.getAnomalyGrade(), 1e-10);
                assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
                // grades will not match between first and third because the thresholder has
                // wrong info
                // about shinglesize
            }
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest fourth = mapper.toModel(mapper.toState(second));
            for (int j = length; j < shingledData.length; j++) {
                // validate eaulity of points
                for (int y = 0; y < baseDimensions; y++) {
                    assertEquals(dataWithKeys.data[count][y], shingledData[j][(shingleSize - 1) * baseDimensions + y],
                            1e-10);
                }

                AnomalyDescriptor firstResult = first.process(dataWithKeys.data[count], 0L);
                ++count;
                AnomalyDescriptor secondResult = second.process(shingledData[j], 0L);
                AnomalyDescriptor thirdResult = third.process(shingledData[j], 0L);
                AnomalyDescriptor fourthResult = fourth.process(shingledData[j], 0L);

                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getAnomalyGrade(), secondResult.getAnomalyGrade(), 1e-10);
                assertEquals(firstResult.getRcfScore(), thirdResult.getRcfScore(), 1e-10);
                // grades will not match between first and third because the thresholder has
                // wrong info
                // about shinglesize
                assertEquals(firstResult.getRcfScore(), fourthResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getAnomalyGrade(), fourthResult.getAnomalyGrade(), 1e-10);

            }
        }
    }

}
