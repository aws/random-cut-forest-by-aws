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

public class AttributionThresholder extends BasicThresholder {

    protected Deviation scoreDiff;
    protected DiVector lastAnomalyAttribution;
    protected boolean ignoreSimilar;

    public AttributionThresholder(double discount, int baseDimension, double absoluteThreshold,int minimumScores, boolean ignoreSimilar){
        super(discount,baseDimension, absoluteThreshold,minimumScores);
        this.ignoreSimilar = ignoreSimilar;
        this.scoreDiff = new Deviation();
    }

    public AttributionThresholder(boolean isInAnomaly, double discount, int count, Deviation simpleDev, int baseDimension, double absoluteThreshold,int minimumScores, double lastScore, double lastAnomalyScore, Deviation scoreDiff, DiVector lastAnomalyAttribution, boolean ignoreSimilar){
        super(isInAnomaly, discount,count, simpleDev, baseDimension, absoluteThreshold,minimumScores,lastScore,lastAnomalyScore);
        this.ignoreSimilar = ignoreSimilar;
        this.scoreDiff = scoreDiff;
        this.lastAnomalyAttribution = new DiVector(lastAnomalyAttribution);
    }

    public int process(double newScore, DiVector attribution, int timeStamp){
        return process(newScore,newScore,attribution,null,timeStamp);
    }

    public int process(double newScore, double idealScore, DiVector attribution, DiVector idealAttribution, int timeStamp) {
        checkArgument(!moreInformation || attribution != null, "incorrect state, need more information");

        final int answer;
        if (isPotentialAnomaly(newScore)) {
            if (attribution == null) {
                    moreInformation = true;
                    return MORE_INFORMATION;
                } else {
                moreInformation = false;
                if (!inAnomaly) {
                    answer = START_OF_ANOMALY;
                    lastAnomalyScore = newScore;
                    lastAnomalyTimeStamp = timeStamp;
                    inAnomaly = true;
                    lastAnomalyAttribution = new DiVector(attribution);
                } else {
                    if (trigger(attribution, timeStamp)) {//|| newScore > lastAnomalyScore + 1.5 * scoreDiff.getDeviation()) {
                        answer = CONTINUED_ANOMALY_HIGHLIGHT;
                        lastAnomalyScore = newScore;
                        lastAnomalyTimeStamp = timeStamp;
                        lastAnomalyAttribution = new DiVector(attribution);
                    } else {
                        answer = CONTINUED_ANOMALY_NOT_A_HIGHLIGHT;
                    }
                }
            }
        } else {
            answer = NOT_ANOMALY;
            inAnomaly = false;
        }
        ++count;
        simpleDeviation.update(newScore);
        if (count>1 && newScore > lastScore){
            scoreDiff.update(newScore - lastScore);
        }
        lastScore = newScore;
        return answer;
    }

    protected boolean trigger(DiVector candidate, int timeStamp) {
        checkArgument(lastAnomalyAttribution.getDimensions() == candidate.getDimensions(), " error in DiVectors");
        int dimensions = candidate.getDimensions();

        int difference = baseDimension * (timeStamp - lastAnomalyTimeStamp);

        if (difference < dimensions) {
            double remainder = 0;
            for (int i = dimensions - difference; i < dimensions; i++) {
                remainder += candidate.getHighLowSum(i);
            }
            return isPotentialAnomaly(remainder * dimensions / difference);
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

    public boolean isIgnoreSimilar() {
        return ignoreSimilar;
    }

    public Deviation getScoreDiff() {
        return scoreDiff;
    }

    public DiVector getLastAnomalyAttribution() {
        return lastAnomalyAttribution;
    }
}
