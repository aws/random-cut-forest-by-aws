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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.examples.datasets.ShingledData;

public class ERCFexample implements Example {

    public static void main(String[] args) throws Exception {
        new ERCFexample().run();
    }

    @Override
    public String command() {
        return "ERCF_example";
    }

    @Override
    public String description() {
        return "ERCF Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_64;
        int dataSize = 4 * sampleSize;
        int baseDimensions = 1;
        int CONFIG_NUMBER_OF_ATTRIBUTORS = 2;

        int dimensions = baseDimensions * shingleSize;
        EnrichedRandomCutForest forest = new EnrichedRandomCutForest(RandomCutForest.builder().compact(true).dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize).precision(precision));


        for (double[] point : ShingledData.generateShingledData(dataSize, 50, dimensions, 0)) {
            AnomalyDescriptor result = forest.process(point);
            if (result != null) {
                System.out.print("timestamp " + result.timeStamp + ", value ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.currentValues[i] + ", ");
                }
                System.out.print("score " + result.score + ", ");


                if (result.startOfAnomaly) {
                    System.out.print(-result.relativeIndex + " steps ago, instead of ");
                    for (int i = 0; i < baseDimensions; i++) {
                        System.out.print(result.oldValues[i] + ", ");
                    }
                }
                System.out.print("expected ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.expectedValuesList[0][i] + ", ");
                }
                System.out.println();
            }
        }

    }

}