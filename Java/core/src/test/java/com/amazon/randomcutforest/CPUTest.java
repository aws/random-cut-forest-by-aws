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

import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;

/**
 * The following "test" is intended to provide an approximate estimate of the improvement
 * from parallelization. At the outset, we remark that running the test from inside
 * an IDE/environment may reflect more of the environment. Issues such as warming are not
 * reflected in this test.
 *
 * Users who wish to obtain more calibrated estimates should use a benchmark -- preferably
 * using their own "typical" data and their end to end setup. Performance of RCF is data dependent.
 * Such users may be invoking RCF functions differently from a standard "impute, score, update"
 * process recommended for streaming time series data.
 *
 * Moreover, in the context of a large number of models, the rate at which the models require
 * updates is also a factor and not controlled herein.
 *
 * The two tests should produce near identical sum of scores, and (root) mean squared error of
 * the impute up to machine precision (since the order of the arithmetic operations would vary).
 *
 * To summarize the lessons, it appears that parallelism almost always helps (upto resource limitations).
 * If an user is considering a single model -- say from a console or dashboard, they should consider
 * having parallel threads enabled. For large number of models, it may be worthwhile
 * to also investigate different ways of achieving parallelism and not just attempt to
 * change the executor framework.
 *
 */

@Tag("functional")
public class CPUTest {

    int numberOfTrees = 30;
    int DATA_SIZE = 10000;
    int numberOfForests = 6;
    int numberOfAttributes = 5;
    int shingleSize = 30;
    int sampleSize = 256;
    // set numberOfThreads = 1 to turn off parallelism
    int numberOfThreads = 3;
    // change boundingBoxCacheFraction to see different memory consumption
    // this would be germane for large number of models cache/memory contention
    double boundingBoxCacheFraction = 1.0;
    int dimensions = shingleSize * numberOfAttributes;

    @Test
    public void profileTestSync() {
        double [] mse = new double [numberOfForests];
        int [] mseCount = new int[numberOfForests];
        double [] score =new double[numberOfForests];

        double[][] data = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0, numberOfAttributes).data;

        RandomCutForest [] forests = new RandomCutForest [numberOfForests];
        for (int k = 0;k<numberOfForests; k++) {
            forests[k] = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).randomSeed(99+k).outputAfter(10)
                    .parallelExecutionEnabled(true)
                    .threadPoolSize(numberOfThreads)
                    .internalShinglingEnabled(true).initialAcceptFraction(0.1).sampleSize(sampleSize).build();
        }

        for (int j = 0; j < data.length; j++) {
            for (int k = 0;k<numberOfForests; k++) {
                score[k] += forests[k].getAnomalyScore(data[j]);
                if (j % 10 == 0 && j > 0) {
                    double[] result = forests[k].extrapolate(1);
                    double sum = 0;
                    for (int i = 0; i < result.length; i++) {
                        double t = result[i] - data[j][i];
                        sum += t * t;
                    }
                    sum = Math.sqrt(sum);
                    mse[k] += sum;
                    mseCount[k]++;
                }
                forests[k].update(data[j]);
            }
        }
        for(int k=0;k<numberOfForests;k++) {
            System.out.println(" Forest " + k);
            System.out.println(" MSE " + mse[k] / mseCount[k]);
            System.out.println(" scoresum " + score[k] / data.length);
        }
    }

    @Test
    public void profileTestASync() {
        double [] mse = new double [numberOfForests];
        int [] mseCount = new int[numberOfForests];
        double [] score =new double[numberOfForests];

        double[][] data = ShingledMultiDimDataWithKeys.getMultiDimData(DATA_SIZE, 60, 100, 5, 0, numberOfAttributes).data;

        RandomCutForest [] forests = new RandomCutForest [numberOfForests];
        for (int k = 0;k<numberOfForests; k++) {
            forests[k] = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions).shingleSize(shingleSize)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).randomSeed(99+k).outputAfter(10)
                    .parallelExecutionEnabled(false)
                    .internalShinglingEnabled(true).initialAcceptFraction(0.1).sampleSize(sampleSize).build();
        }

        ForkJoinPool forkJoinPool = new ForkJoinPool(numberOfThreads);
        int [] indices = new int[numberOfForests];
        for(int k=0;k<numberOfForests;k++){
            indices[k] = k;
        }

        for (int j = 0; j < data.length; j++) {
            int finalJ=j;
            forkJoinPool.submit( () ->
            Arrays.stream(indices).parallel().forEach(k -> {
                score[k] += forests[k].getAnomalyScore(data[finalJ]);
                if (finalJ % 10 == 0 && finalJ > 0) {
                    double[] result = forests[k].extrapolate(1);
                    double sum = 0;
                    for (int i = 0; i < result.length; i++) {
                        double t = result[i] - data[finalJ][i];
                        sum += t * t;
                    }
                    sum = Math.sqrt(sum);
                    mse[k] += sum;
                    mseCount[k]++;
                }
                forests[k].update(data[finalJ]);
            })).join();
        }
        for(int k=0;k<numberOfForests;k++) {
            System.out.println(" Forest " + k);
            System.out.println(" MSE " + mse[k] / mseCount[k]);
            System.out.println(" scoresum " + score[k] / data.length);
        }
    }

}