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

import static com.amazon.randomcutforest.threshold.BasicThresholder.CONTINUED_ANOMALY_HIGHLIGHT;
import static com.amazon.randomcutforest.threshold.BasicThresholder.START_OF_ANOMALY;

public class CorrectorThresholding implements Example {

    public static void main(String[] args) throws Exception {
        new CorrectorThresholding().run();
    }

    @Override
    public String command() {
        return "corrector_thresholding";
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
        int CONFIG_NUMBER_OF_ATTRIBUTORS=2;

        int dimensions = baseDimensions*shingleSize;
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize).precision(precision).build();

        CorrectorThresholder correctorThresholder = new CorrectorThresholder(1.0/sampleSize,1,true,1.0,10,true);

        double score;
        int count = 0;
        int [] missingIndices;
        int [] likelyMissingIndices = new int [baseDimensions];
        for(int i=0;i<baseDimensions;i++){
            likelyMissingIndices[i] = dimensions - baseDimensions + i;
        }

        for (double[] point : ShingledData.generateShingledData(dataSize, 50, dimensions, 0)) {
            score = forest.getAnomalyScore(point);

            if (score > 0) {
                if (correctorThresholder.process(score) == BasicThresholder.MORE_INFORMATION) {
                    /**
                     * we consider what the most recent values should have been, reflected in newPoint;
                     * and then pass the score, score of newPoint, attribution and attribution of newPoint
                     * to the thresholder. The idea is that "if the most likely least anomalous score is high"
                     * (reflected in forest.getAnomalyScore(newPoint) then the most recent observations are not
                     * an anomaly. If the still are considered an anomaly, then we look at the most egregious
                     * subobservations in the shingle, given by maxContribution() and predict those values
                     * -- note that this may correspond to anomalies being detecting late; and deciding on the
                     * values when we detect anomalies (based on what we know now, as opposed to pure forecasting)
                     *
                     * The parameter CONFIG_NUMBER_OF_ATTRIBUTORS determines the maximum number of different attributors
                     * we could consider; note that larger number of contributors are difficult to visualize/control
                     */
                    DiVector attribution = forest.getAnomalyAttribution(point);
                    int startPosition = attribution.getDimensions() - baseDimensions;
                    likelyMissingIndices = largestFeatures(attribution,startPosition,baseDimensions,CONFIG_NUMBER_OF_ATTRIBUTORS);
                    double[] newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
                    DiVector newAttribution = forest.getAnomalyAttribution(newPoint);
                    int signal = correctorThresholder.process(score, forest.getAnomalyScore(newPoint), attribution, newAttribution, count);

                    if (signal == START_OF_ANOMALY || signal == CONTINUED_ANOMALY_HIGHLIGHT) {
                        System.out.print("timestamp " + count + ", value ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(point[dimensions - baseDimensions + i] + ", ");
                        }
                        System.out.print("score " + score + ", ");

                        int index = maxContribution(attribution, baseDimensions);
                        if (index + 1 < 0 && signal == START_OF_ANOMALY) {
                            // anomaly in the past and detected late
                            startPosition = attribution.getDimensions() + index * baseDimensions;
                            missingIndices = largestFeatures(attribution, startPosition, baseDimensions, CONFIG_NUMBER_OF_ATTRIBUTORS);
                            for (int i = 0; i < baseDimensions; i++) {
                                missingIndices[i] = dimensions + index * baseDimensions + i;
                            }
                            newPoint = forest.imputeMissingValues(point, missingIndices.length, missingIndices);

                            System.out.print(-(index + 1) + " steps ago, instead of ");
                            for (int i = 0; i < baseDimensions; i++) {
                                System.out.print(point[startPosition + i] + ", ");
                            }
                        }
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(newPoint[startPosition + i] + ", ");
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

    private int[] largestFeatures(DiVector diVector, int position, int baseDimension, int max_number){
        double sum = 0;
        for(int i=0;i<baseDimension;i++){
            sum += diVector.getHighLowSum(i + position);
        }
        int count = 0;
        for(int i=0;i<baseDimension;i++){
            if (diVector.getHighLowSum(i + position) > sum/(max_number + 1)) {
                ++count;
            }
        }
        int [] answer = new int [count];
        count = 0;
        for(int i=0;i<baseDimension;i++){
            if (diVector.getHighLowSum(i + position) > sum/(max_number + 1)) {
                answer[count++] = i + position;
            }
        }
        return answer;
    }
}
