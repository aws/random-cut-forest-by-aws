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

    protected boolean attributionEnabled;

    public CorrectorThresholder(double discount, int baseDimension, boolean attributionEnabled, double absoluteThreshold,int minimumScores, boolean ignoreSimilar){
        super(discount,baseDimension, absoluteThreshold,minimumScores, ignoreSimilar);
        this.attributionEnabled = attributionEnabled;
    }

    public CorrectorThresholder(boolean isInAnomaly, double discount, int count, Deviation simpleDev, int baseDimension, double absoluteThreshold,int minimumScores, double lastScore, double lastAnomalyScore, boolean attributionEnabled, Deviation scoreDiff, DiVector lastAnomalyAttribution, boolean ignoreSimilar){
        super(isInAnomaly,discount,count, simpleDev,baseDimension,absoluteThreshold,minimumScores,lastScore,lastAnomalyScore,scoreDiff,lastAnomalyAttribution,ignoreSimilar);
        this.attributionEnabled = attributionEnabled;
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
                    if (!inAnomaly) {
                        answer = START_OF_ANOMALY;
                        lastAnomalyScore = newScore;
                        inAnomaly = true;
                        lastAnomalyAttribution = new DiVector(attribution);
                        lastAnomalyTimeStamp = timeStamp;
                    } else {
                        if (trigger(attribution, timeStamp) && newScore > idealScore + 1.5 * scoreDiff.getDeviation()) {
                            answer = CONTINUED_ANOMALY_HIGHLIGHT;
                            lastAnomalyScore = newScore;
                            lastAnomalyAttribution = new DiVector(attribution);
                            lastAnomalyTimeStamp = timeStamp;
                        } else {
                            answer = CONTINUED_ANOMALY_NOT_A_HIGHLIGHT;
                        }
                    }
                }
            } else {
                if (!inAnomaly) {
                    answer = START_OF_ANOMALY;
                    lastAnomalyScore = newScore;
                    inAnomaly = true;
                } else {
                    if (newScore > idealScore + 1.5 * scoreDiff.getDeviation()) {
                        answer = CONTINUED_ANOMALY_HIGHLIGHT;
                        lastAnomalyScore = newScore;
                        lastAnomalyTimeStamp = timeStamp;
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

    public boolean isAttributionEnabled() {
        return attributionEnabled;
    }
}
