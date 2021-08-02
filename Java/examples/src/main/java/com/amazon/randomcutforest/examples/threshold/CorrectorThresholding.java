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
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.threshold.BasicThresholder;
import com.amazon.randomcutforest.threshold.CorrectorThresholder;

public class CorrectorThresholding implements Example {

    public static void main(String[] args) throws Exception {
        new CorrectorThresholding().run();
    }

    @Override
    public String command() {
        return "highlight_thresholding";
    }

    @Override
    public String description() {
        return "check highlight thresholding";
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

        CorrectorThresholder correctorThresholder = new CorrectorThresholder(1.0/sampleSize,1,true,1.0,10,true);

        double score;
        int count = 0;
        int [] missingIndices = new int [baseDimensions];
        int [] likelyMissingIndices = new int [baseDimensions];
        for(int i=0;i<baseDimensions;i++){
            likelyMissingIndices[i] = dimensions - baseDimensions + i;
        }

        for (double[] point : ShingledData.generateShingledData(dataSize, 50, dimensions, 0)) {
            score = forest.getAnomalyScore(point);
            int signal = 0;
            int index = 0;
            if (score > 0) {
                signal = correctorThresholder.process(score);
                if (signal < 0) {
                    DiVector attribution = forest.getAnomalyAttribution(point);
                    double[] newPoint = forest.imputeMissingValues(point, baseDimensions, likelyMissingIndices);
                    DiVector newAttribution = forest.getAnomalyAttribution(newPoint);
                    signal = correctorThresholder.process(score, forest.getAnomalyScore(newPoint), attribution, newAttribution, count);
                    if (signal == BasicThresholder.CONTINUED_ANOMALY_HIGHLIGHT) {
                        System.out.print(count + " value ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(point[dimensions - baseDimensions + i] + ", ");
                        }
                        index = maxContribution(attribution, baseDimensions);
                        for (int i = 0; i < baseDimensions; i++) {
                            missingIndices[i] = dimensions + index * baseDimensions + i;
                        }
                        double[] expected = forest.imputeMissingValues(point, baseDimensions, missingIndices);
                        System.out.print("score " + score + ", ");
                        if (index + 1 < 0){
                            System.out.print( -(index+1) + " steps ago, instead of ");
                            for(int i=0;i<baseDimensions;i++) {
                                System.out.print(point[missingIndices[i]] + ", ");
                            }
                        }
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(expected[missingIndices[i]] + ", ");
                        }
                        System.out.println();
                    }

                }
            }
            ++count;
            forest.update(point);
        }

    }

    private int maxContribution(DiVector diVector, int baseDimension){
        double val = 0;
        for(int i = 0;i<baseDimension;i++){
            val += diVector.getHighLowSum(i);
        }
        int index = - diVector.getDimensions()/baseDimension;
        for(int i = baseDimension;i<diVector.getDimensions();i += baseDimension){
            double sum = 0;
            for(int j = 0;j<baseDimension;j++){
                sum += diVector.getHighLowSum(i + j);
            }
            if (sum>val){
                val = sum;
                index = (i-diVector.getDimensions())/baseDimension;
            }
        }
        return index;
    }
}
