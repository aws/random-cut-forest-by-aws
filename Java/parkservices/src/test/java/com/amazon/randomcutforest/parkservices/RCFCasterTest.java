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

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RCFCasterTest {

    @Test
    void testRCFCast() {
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 10 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 1;
        int forecastHorizon = 15;
        int shingleSize = 20;
        int outputAfter = 32;
        int errorHorizon = 256;

        long seed = new Random().nextLong();

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                50, 5, seed, baseDimensions, false);

        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NORMALIZE;
        RCFCaster caster = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .errorHorizon(errorHorizon).initialAcceptFraction(0.125).build();
        RCFCaster shadow = RCFCaster.builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
                .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
                .errorHorizon(errorHorizon).initialAcceptFraction(0.125).boundingBoxCacheFraction(0).build();

        // ensuring that the parameters are the same; otherwise the grades/scores cannot
        // be the same
        // weighTime has to be 0
        caster.setLowerThreshold(1.1);
        shadow.setLowerThreshold(1.1);

        assert (caster.errorHandler.errorHorizon == errorHorizon);
        assert (caster.errorHorizon == errorHorizon);
        double score = 0;
        for (int j = 0; j < dataWithKeys.data.length; j++) {
            ForecastDescriptor result = caster.process(dataWithKeys.data[j], 0L);
            ForecastDescriptor shadowResult = shadow.process(dataWithKeys.data[j], 0L);
            assertArrayEquals(shadowResult.getTimedForecast().rangeVector.values,
                    result.getTimedForecast().rangeVector.values, 1e-6f);
            score += result.getRCFScore();
            int sequenceIndex = caster.errorHandler.sequenceIndex;
            if (caster.forest.isOutputReady()) {
                float[] meanArray = caster.errorHandler.getErrorMean();
                for (int i = 0; i < forecastHorizon; i++) {
                    int len = (sequenceIndex > errorHorizon + i + 1) ? errorHorizon : sequenceIndex - i - 1;
                    if (len > 0) {
                        for (int k = 0; k < baseDimensions; k++) {
                            int pos = i * baseDimensions + k;
                            double[] array = caster.errorHandler.getErrorVector(len, (i + 1), k, pos,
                                    RCFCaster.defaultError);
                            double mean = Arrays.stream(array).sum() / len;
                            assertEquals(meanArray[pos], mean, (1 + Math.abs(mean)) * 1e-4);
                        }
                    }
                }
                float[] intervalPrecision = shadow.errorHandler.getIntervalPrecision();
                for (float y : intervalPrecision) {
                    assertTrue(0 <= y && y <= 1.0);
                }
                assertArrayEquals(intervalPrecision, result.getIntervalPrecision());
            }

        }

    }

}
