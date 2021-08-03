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

package com.amazon.randomcutforest.examples.threshold;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.examples.datasets.ShingledData;
import com.amazon.randomcutforest.threshold.BasicThresholder;

public class SimpleThresholding implements Example {

    public static void main(String[] args) throws Exception {
        new SimpleThresholding().run();
    }

    @Override
    public String command() {
        return "simple_thresholding";
    }

    @Override
    public String description() {
        return "check simple thresholding";
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

        int dimensions = baseDimensions*shingleSize;
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize).precision(precision).build();

        BasicThresholder simpleThresholder = new BasicThresholder(1.0/sampleSize,1,1.0,10);

        double score;
        int count = 0;
        int [] missingIndices = new int [baseDimensions];
        for(int i=0;i<baseDimensions;i++){
            missingIndices[i] = dimensions - baseDimensions + i;
        }

        for (double[] point : ShingledData.generateShingledData(dataSize, 50, shingleSize, 0)) {
            score = forest.getAnomalyScore(point);
            if (score > 0){
                if (simpleThresholder.process(score) == BasicThresholder.IS_ANOMALY) {
                    /**
                     * Note that in the simplest case, we just assume that the anomaly designation is
                     * due to the most recent observations.
                     */
                    System.out.print(count + " value ");
                    for(int i=0;i<baseDimensions;i++) {
                        System.out.print(point[dimensions - baseDimensions + i] + ", ");
                    }
                    double [] expected = forest.imputeMissingValues(point,baseDimensions,missingIndices);
                    System.out.print("score " + score + " expected ");
                    for(int i=0;i<baseDimensions;i++) {
                        System.out.print(expected[dimensions - baseDimensions + i] + ", ");
                    }
                    System.out.println();
                }
            }
            ++count;
            forest.update(point);
        }

    }
}
