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
import com.amazon.randomcutforest.threshold.AttributionThresholder;
import com.amazon.randomcutforest.threshold.BasicThresholder;

public class AttributionThresholding implements Example {

    public static void main(String[] args) throws Exception {
        new AttributionThresholding().run();
    }

    @Override
    public String command() {
        return "attribution_thresholding";
    }

    @Override
    public String description() {
        return "check attribution thresholding";
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


        AttributionThresholder highlightThresholder = new AttributionThresholder(1.0/sampleSize,1,true,1.0,10,true);

        double score;
        int count = 0;
        int [] missingIndices = new int [baseDimensions];

        for (double[] point : ShingledData.generateShingledData(dataSize, 50, dimensions, 0)) {
            score = forest.getAnomalyScore(point);
            int index = 0;
            if (score > 0){
                if (highlightThresholder.process(score) == BasicThresholder.MORE_INFORMATION){
                    /**
                     * The goal is to chek if the score is high simply because an already detected anomaly
                     * is in the shingle. If not, then we consider the most egregeous observations in the
                     * shingle given by maxContribution() and impute the values -- note that anonalies may be
                     * detected late and this is not pure forecasting.
                     */
                    DiVector attribution = forest.getAnomalyAttribution(point);
                    int signal = highlightThresholder.process(score, attribution, count);
                    index = maxContribution(attribution,baseDimensions);
                    if (signal == BasicThresholder.CONTINUED_ANOMALY_HIGHLIGHT) {
                        System.out.print(count + " value ");
                        for(int i=0;i<baseDimensions;i++) {
                            System.out.print(point[dimensions - baseDimensions + i] + ", ");
                        }
                        for(int i=0;i<baseDimensions;i++){
                            missingIndices[i] = dimensions + index*baseDimensions + i;
                        }
                        double [] expected = forest.imputeMissingValues(point,baseDimensions,missingIndices);
                        System.out.print("score " + score + ", ");
                        if (index + 1 < 0){
                            System.out.print( -(index+1) + " steps ago, instead of ");
                            for(int i=0;i<baseDimensions;i++) {
                                System.out.print(point[missingIndices[i]] + ", ");
                            }
                        }
                        System.out.print("expected ");
                        for(int i=0;i<baseDimensions;i++) {
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
