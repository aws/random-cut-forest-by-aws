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

package com.amazon.randomcutforest.examples.ERCF;

import com.amazon.randomcutforest.ERCF.AnomalyDescriptor;
import com.amazon.randomcutforest.ERCF.ExtendedRandomCutForest;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.examples.datasets.MultiDimDataWithKey;
import com.amazon.randomcutforest.examples.datasets.ShingledMultiDimDataWithKeys;

import java.util.Arrays;
import java.util.Random;

public class ERCF_MultiDimensionalExample implements Example {

    public static void main(String[] args) throws Exception {
        new ERCF_MultiDimensionalExample().run();
    }

    @Override
    public String command() {
        return "ERCF_Multi_Dim_example";
    }

    @Override
    public String description() {
        return "ERCF Multi Dimensional Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // note that the number of anomalies are 1% per dimension;
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 5;

        int dimensions = baseDimensions * shingleSize;
        ExtendedRandomCutForest forest = new ExtendedRandomCutForest(RandomCutForest.builder().compact(true)
                .dimensions(dimensions).randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize)
                .sampleSize(sampleSize).outputAfter(32).precision(precision), 0.01);

        long seed = new Random().nextLong();
        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.generateShingledDataWithKey(dataSize, 50, shingleSize, baseDimensions,
                seed);
        int keyCounter = 0;
        for (double[] point : dataWithKeys.data) {

            AnomalyDescriptor result = forest.process(point);

            if (keyCounter < dataWithKeys.changeIndices.length
                    && result.timeStamp + shingleSize - 1 == dataWithKeys.changeIndices[keyCounter]){
                System.out.println("timestamp " + (result.timeStamp + shingleSize - 1) + " CHANGE " + Arrays.toString(dataWithKeys.changes[keyCounter]));
                ++keyCounter;
            }

            if (result.anomalyGrade != 0) {
                System.out.print("timestamp " + (result.timeStamp + shingleSize - 1) + " RESULT value ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.currentValues[i] + ", ");
                }
                System.out.print("score " + result.rcfScore + ", grade " + result.anomalyGrade + ", ");

                if (result.expectedValuesPresent) {
                    if (result.relativeIndex != 0 && result.startOfAnomaly) {
                        System.out.print(-result.relativeIndex + " steps ago, instead of ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.oldValues[i] + ", ");
                        }
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.expectedValuesList[0][i] + ", ");
                            if (result.oldValues[i] != result.expectedValuesList[0][i]) {
                                System.out.print("( " + (result.oldValues[i] - result.expectedValuesList[0][i]) + " ) ");
                            }
                        }
                    } else {
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.expectedValuesList[0][i] + ", ");
                            if (result.currentValues[i] != result.expectedValuesList[0][i]) {
                                System.out.print("( " + (result.currentValues[i] - result.expectedValuesList[0][i]) + " ) ");
                            }
                        }
                    }
                }
                System.out.println();
            }
        }

    }

}