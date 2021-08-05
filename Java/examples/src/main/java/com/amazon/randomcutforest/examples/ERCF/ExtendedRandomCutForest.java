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
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.threshold.BasicThresholder;
import com.amazon.randomcutforest.threshold.CorrectorThresholder;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class ExtendedRandomCutForest{

    protected CorrectorThresholder correctorThresholder;

    protected RandomCutForest forest;

    protected int count = 0;

    int baseDimensions;

    int shingleSize;

    // for anomaly description we would only look at these may top attributors
    // note that expected value is not well defined when this number is greater than 1
    public static int CONFIG_NUMBER_OF_ATTRIBUTORS = 2;

    // the number of scores we should see before vending judgements of anomaly/anomaly
    // more is better, the setting below is an optimistic setting
    public static int MINIMUM_SCORES = 10;

    // the score below which we will ignore anomalies; there can be examples where the setting of 1.0
    // is large -- but the setting below (and increasing it) should reduce the number of anomalies flagged.
    public static double ABSOLUTE_THRESHOLD = 1.0;


    public ExtendedRandomCutForest(RandomCutForest.Builder<?> builder, double anomalyRate) {
        forest = new RandomCutForest(builder);
        checkArgument(!forest.isInternalShinglingEnabled(), "internal shingling not suported");
        baseDimensions = forest.getDimensions() / forest.getShingleSize();
        shingleSize = forest.getShingleSize();
        correctorThresholder = new CorrectorThresholder(anomalyRate, baseDimensions, true, 0.8, MINIMUM_SCORES, false);
    }

    public ExtendedRandomCutForest(RandomCutForest forest, CorrectorThresholder thresholder, int count){
        checkArgument(!forest.isInternalShinglingEnabled(), "internal shingling not suported");
        this.forest = forest;
        baseDimensions = forest.getDimensions() / forest.getShingleSize();
        shingleSize = forest.getShingleSize();
        this.correctorThresholder = thresholder;
        this.count = count;
    }

    public AnomalyDescriptor process(double[] point) {
        AnomalyDescriptor result = new AnomalyDescriptor();
        result.score = forest.getAnomalyScore(point);
        if (result.score <= 0){
            result.anomalyGrade = 0;
        } else {
             if (correctorThresholder.process(result.score) == BasicThresholder.MORE_INFORMATION) {
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
                int [] likelyMissingIndices = largestFeatures(attribution, startPosition, baseDimensions, CONFIG_NUMBER_OF_ATTRIBUTORS);
                double[] newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
                DiVector newAttribution = forest.getAnomalyAttribution(newPoint);
                int signal = correctorThresholder.process(result.score, forest.getAnomalyScore(newPoint), attribution, newAttribution, count);

                if (signal>0) {
                    result.anomalyGrade = 0.01*signal;
                    result.attribution = new DiVector(attribution);
                    result.timeStamp = count;
                    result.currentValues = new double[baseDimensions];
                    for(int i = 0;i<baseDimensions;i++){
                        result.currentValues[i] = point[startPosition + i];
                    }
                    result.relativeIndex = maxContribution(attribution, baseDimensions,-shingleSize) + 1;

                    if (result.relativeIndex < 0) {
                        // anomaly in the past and detected late
                        startPosition = attribution.getDimensions() + (result.relativeIndex - 1) * baseDimensions;
                        int [] missingIndices = largestFeatures(attribution, startPosition, baseDimensions, CONFIG_NUMBER_OF_ATTRIBUTORS);
                        newPoint = forest.imputeMissingValues(point, missingIndices.length, missingIndices);
                        result.oldValues = new double[baseDimensions];
                        result.flattenedAttribution = new double[baseDimensions];
                        for (int i = 0; i < baseDimensions; i++) {
                            result.oldValues[i] = point[startPosition + i];
                            result.flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
                        }
                    }
                    result.expectedValuesList = new double [1] [];
                    result.expectedValuesList[0] = new double[baseDimensions];
                    result.flattenedAttribution = new double[baseDimensions];
                    for (int i = 0; i < baseDimensions; i++) {
                        result.expectedValuesList[0][i] = newPoint[startPosition + i];
                        result.flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
                    }
                    result.likelihoodOfValues = new double [] {1.0};
                } else {
                    result.anomalyGrade = 0;
                }
            }
        }
        ++count;
        forest.update(point);
        return result;
    }


    private int maxContribution(DiVector diVector, int baseDimension, int startIndex) {
        double val = 0;
        int index = startIndex;
        int position = diVector.getDimensions() + startIndex*baseDimension;
        for (int i = 0; i < baseDimension; i++) {
            val += diVector.getHighLowSum(i + position);
        }
        for (int i = position + baseDimension; i < diVector.getDimensions(); i += baseDimension) {
            double sum = 0;
            for (int j = 0; j < baseDimension; j++) {
                sum += diVector.getHighLowSum(i + j);
            }
            if (sum > val) {
                val = sum;
                index = (i - diVector.getDimensions()) / baseDimension;
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

    public RandomCutForest getForest() {
        return forest;
    }

    public CorrectorThresholder getCorrectorThresholder() {
        return correctorThresholder;
    }

    public int getCount() {
        return count;
    }
}
