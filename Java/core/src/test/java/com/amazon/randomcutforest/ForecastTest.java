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

package com.amazon.randomcutforest;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys.generateShingledData;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag("functional")
public class ForecastTest {

    @Test
    public void basic() {

        int sampleSize = 256;
        int baseDimensions = 1;
        int shingleSize = 8;
        int dimensions = baseDimensions * shingleSize;
        long seed = 8008895784411556298L;
        new Random().nextLong();
        System.out.println(seed);

        int length = 4 * sampleSize;
        int outputAfter = 128;

        RandomCutForest forest = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true).shingleSize(shingleSize)
                .outputAfter(outputAfter).build();

        // as the ratio of amplitude (signal) to noise is changed, the estimation range
        // in forecast
        // (or any other inference) should increase
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 10, seed,
                baseDimensions);

        System.out.println(dataWithKeys.changes.length + " anomalies injected ");
        double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, false);

        assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

        int horizon = 20;
        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            // forecast first; change centrality to achieve a control over the sampling
            // setting centrality = 0 would correspond to random sampling from the leaves
            // reached by
            // impute visitor
            RangeVector forecast = forest.extrapolateFromShingle(forest.lastShingledPoint(), horizon, 1, 1.0);
            assert (forecast.values.length == horizon);
            for (int i = 0; i < horizon; i++) {
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
            forest.update(dataWithKeys.data[j]);
        }

        System.out.println("RMSE ");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower ");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper ");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

}
