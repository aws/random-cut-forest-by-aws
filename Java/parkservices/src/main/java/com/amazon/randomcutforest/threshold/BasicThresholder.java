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


import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import com.amazon.randomcutforest.returntypes.DiVector;

public class BasicThresholder {

    public static int IS_ANOMALY = 1;

    public static int NOT_ANOMALY = 0;

    public static int MORE_INFORMATION = -1;

    protected boolean moreInformation;

    public static int CONTINUED_ANOMALY_NOT_A_HIGHLIGHT = 2;

    public static int CONTINUED_ANOMALY_HIGHLIGHT = 3;

    protected boolean inAnomaly = false;

    protected double elasticity = 0.01;

    protected boolean attributionEnabled  = false;

    protected int count = 0;

    protected double discount = 0;

    protected int baseDimension = 1;

    protected int minumumScores = 0;

    protected Deviation simpleDeviation;

    protected int lastAnomalyTimeStamp;

    protected double absoluteThreshold = Double.MAX_VALUE;

    protected double lastAnomalyScore;

    protected DiVector lastAnomalyAttribution;

    public BasicThresholder(){
        simpleDeviation = new Deviation();
    }

    public BasicThresholder(double discount){
        simpleDeviation = new Deviation(discount);
        this.discount = discount;
    }

    public BasicThresholder(double discount, int baseDimension, boolean attributionEnabled, double absoluteThreshold, int minumumScores){
        this(discount);
        this.baseDimension = baseDimension;
        this.attributionEnabled = attributionEnabled;
        this.absoluteThreshold = absoluteThreshold;
        this.minumumScores = minumumScores;
        moreInformation = false;
    }

    protected double basicThreshold(){
        return simpleDeviation.getMean() + 3 * simpleDeviation.getDeviation();
    }

    protected boolean isPotentialAnomaly(double newScore){
        // cannot change any state

        if (count <= minumumScores){
            return false;
        }


        if (newScore < absoluteThreshold) {
            return false;
        }


        if (inAnomaly) {
            return (newScore > basicThreshold() - elasticity);
        } else {
            return (newScore > basicThreshold() + elasticity);
        }
    }

    public int process(double newScore){
        return process(newScore,newScore, null, null, 0);
    }

    public int process(double newScore, double idealScore, DiVector attribution, DiVector idealAttrib, int timeStamp) {
        checkArgument(!moreInformation || attribution != null, "incorrect state, need more information");

        final int answer;
        if (isPotentialAnomaly(newScore)) {
                if (attributionEnabled) {
                    if (attribution == null) {
                        moreInformation = true;
                        return MORE_INFORMATION;
                    } else {
                        moreInformation = false;
                        lastAnomalyAttribution = new DiVector(attribution);
                    }
                }
                lastAnomalyScore = newScore;
                lastAnomalyTimeStamp = timeStamp;
                inAnomaly = true;
                answer = IS_ANOMALY;
        } else {
                answer = NOT_ANOMALY;
                inAnomaly = false;
        }
        ++count;
        simpleDeviation.update(newScore);
        return answer;
    }

}
