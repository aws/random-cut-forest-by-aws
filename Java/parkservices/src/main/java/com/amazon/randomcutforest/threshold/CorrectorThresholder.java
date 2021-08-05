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

package com.amazon.randomcutforest.threshold;


import com.amazon.randomcutforest.returntypes.DiVector;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class CorrectorThresholder extends AttributionThresholder {

    public static double UPPER_ANCHOR_SCORE = 2.0;
    public static double Z_FACTOR = 2.5;

    protected boolean attributionEnabled;
    protected boolean previousIsPotentialAnomaly;

    public CorrectorThresholder(double discount, int baseDimension, boolean attributionEnabled, double absoluteThreshold,int minimumScores, boolean ignoreSimilar){
        super(discount,baseDimension, absoluteThreshold,minimumScores, ignoreSimilar);
        this.attributionEnabled = attributionEnabled;
    }

    public CorrectorThresholder(boolean isInAnomaly, double discount, int count, Deviation simpleDev, int baseDimension, double absoluteThreshold,int minimumScores, double lastScore, double lastAnomalyScore, boolean attributionEnabled, Deviation scoreDiff, DiVector lastAnomalyAttribution, boolean ignoreSimilar,boolean previous){
        super(isInAnomaly,discount,count, simpleDev,baseDimension,absoluteThreshold,minimumScores,lastScore,lastAnomalyScore,scoreDiff,lastAnomalyAttribution,ignoreSimilar);
        this.attributionEnabled = attributionEnabled;
        previousIsPotentialAnomaly = previous;
    }

    public int process(double newScore, double idealScore, int timeStamp){
        checkArgument(!attributionEnabled, "need attribution information");
        return process(newScore,idealScore,null,null,timeStamp);
    }

    public int process(double newScore, double idealScore, DiVector attribution, DiVector idealAttribution, int timeStamp) {
        checkArgument(!moreInformation || attribution != null, "incorrect state, need more information");

        final int answer;
        if (isPotentialAnomaly(newScore)) {
            if (attributionEnabled) {
                if (attribution == null) {
                    moreInformation = true;
                    return MORE_INFORMATION;
                } else {
                    moreInformation = false;
                    if (!inAnomaly && trigger(attribution,timeStamp)) {
                        answer = getFactor(newScore);
                        lastAnomalyScore = newScore;
                        inAnomaly = true;
                        lastAnomalyAttribution = new DiVector(attribution);
                        lastAnomalyTimeStamp = timeStamp;
                        if (useLastScore()){
                            scoreDiff.update(Math.max(0,Math.min(UPPER_ANCHOR_SCORE,newScore) - lastScore));
                        }
                    } else {
                        if (trigger(attribution, timeStamp) && newScore > idealScore + Z_FACTOR * scoreDiff.getDeviation()) {
                            answer = getFactor(newScore);
                            lastAnomalyScore = newScore;
                            lastAnomalyAttribution = new DiVector(attribution);
                            lastAnomalyTimeStamp = timeStamp;
                        } else {
                            // this is to enable easier grade computation
                            answer = NOT_ANOMALY;
                        }
                    }
                }
            } else {
                if (!inAnomaly) {
                    answer = getFactor(newScore);
                    lastAnomalyScore = newScore;
                    inAnomaly = true;
                    if (useLastScore()){
                        scoreDiff.update(Math.min(2,newScore) - lastScore);
                    }
                } else {
                    if (newScore > idealScore + 1.5 * scoreDiff.getDeviation()) {
                        answer = getFactor(newScore);
                        lastAnomalyScore = newScore;
                        lastAnomalyTimeStamp = timeStamp;
                    } else {
                        // easier for grade
                        answer = NOT_ANOMALY;
                    }
                }
            }
            previousIsPotentialAnomaly = true;
        } else {
            answer = NOT_ANOMALY;
            inAnomaly = false;
            if (useLastScore()){
                scoreDiff.update(newScore - lastScore);
            }
            previousIsPotentialAnomaly = false;
        }
        ++count;
        simpleDeviation.update(newScore);
        lastScore = newScore;
        return answer;
    }



    @Override
    protected boolean trigger(DiVector candidate, int timeStamp) {
        if (lastAnomalyAttribution == null) {
            return true;
        }
        checkArgument(lastAnomalyAttribution.getDimensions() == candidate.getDimensions(), " error in DiVectors");
        int dimensions = candidate.getDimensions();

        int difference = baseDimension * (timeStamp - lastAnomalyTimeStamp);

        if (difference < dimensions) {
            double remainder = 0;
            for (int i = dimensions - difference; i < dimensions; i++) {
                remainder += candidate.getHighLowSum(i);
            }
            return (remainder * dimensions / difference > conservativeThreshold());
        } else {
            if (!ignoreSimilar) {
                return true;
            }
            double sum = 0;
            for (int i=0;i<dimensions;i++){
                sum += Math.abs(lastAnomalyAttribution.high[i] - candidate.high[i]) +
                        Math.abs(lastAnomalyAttribution.low[i] - candidate.low[i]);
            }
            return (sum > 0.3*lastScore);
        }
    }


    protected boolean useLastScore(){
        return count>0 && !previousIsPotentialAnomaly;
    }

    @Override
    public double basicThreshold(){
        return simpleDeviation.getMean() + Z_FACTOR*(scoreDiff.getDeviation());
    }

    int getFactor(double score){
        return 100 - (int) (100 * scoreDiff.getDeviation()/(score - simpleDeviation.getMean()));
    }

    protected double conservativeThreshold(){
        return simpleDeviation.getMean() + 3 * simpleDeviation.getDeviation();
    }

    public boolean isAttributionEnabled() {
        return attributionEnabled;
    }

    public boolean isPreviousIsPotentialAnomaly() {
        return previousIsPotentialAnomaly;
    }
}
