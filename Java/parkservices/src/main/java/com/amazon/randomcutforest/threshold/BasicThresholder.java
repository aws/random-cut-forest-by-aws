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

public class BasicThresholder {

    public static int IS_ANOMALY = 1;

    public static int NOT_ANOMALY = 0;

    public static int MORE_INFORMATION = -1;

    protected boolean moreInformation;

    public static int START_OF_ANOMALY = 4;

    public static int CONTINUED_ANOMALY_NOT_A_HIGHLIGHT = 2;

    public static int CONTINUED_ANOMALY_HIGHLIGHT = 3;

    protected boolean inAnomaly = false;

    protected double elasticity = 0.01;

    protected int count = 0;

    protected double discount;

    protected int baseDimension;

    protected int minimumScores;

    protected Deviation simpleDeviation;

    protected int lastAnomalyTimeStamp;

    protected double absoluteThreshold;

    protected double lastAnomalyScore;

    protected double lastScore;

    public BasicThresholder(double discount, int baseDimension, double absoluteThreshold, int minimumScores){
        this(false, discount, 0, new Deviation(discount), baseDimension, absoluteThreshold, minimumScores,-1.0,-1.0);
    }

    public BasicThresholder(boolean isInAnomaly, double discount, int count, Deviation deviation, int baseDimension, double absoluteThreshold, int minimumScores, double lastScore, double lastAnomalyScore){
        this.inAnomaly = isInAnomaly;
        this.discount = discount;
        this.count = count;
        this.simpleDeviation = deviation;
        this.baseDimension = baseDimension;
        this.absoluteThreshold = absoluteThreshold;
        this.minimumScores = minimumScores;
        this.lastAnomalyScore = lastAnomalyScore;
        this.lastScore = lastScore;
        moreInformation = false;
    }

    public double basicThreshold(){
        return simpleDeviation.getMean() + 3 * simpleDeviation.getDeviation();
    }

    public boolean isPotentialAnomaly(double newScore){
        // cannot change any state

        if (count <= minimumScores){
            return false;
        }


        if (newScore < absoluteThreshold) {
            return false;
        }


        if (inAnomaly) {
            return (newScore > basicThreshold() - elasticity);
        } else {
            return (newScore > basicThreshold());
        }
    }

    public int process(double newScore){
        return process(newScore,newScore, null, null, 0);
    }

    public int process(double newScore, double idealScore, DiVector attribution, DiVector idealAttrib, int timeStamp) {
        checkArgument(!moreInformation || attribution != null, "incorrect state, need more information");

        final int answer;
        if (isPotentialAnomaly(newScore)) {
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
        lastScore = newScore;
        return answer;
    }

    public boolean isInAnomaly() {
        return inAnomaly;
    }

    public Deviation getSimpleDeviation() {
        return simpleDeviation;
    }

    public double getAbsoluteThreshold() {
        return absoluteThreshold;
    }

    public double getDiscount() {
        return discount;
    }

    public double getElasticity() {
        return elasticity;
    }

    public double getLastAnomalyScore() {
        return lastAnomalyScore;
    }

    public double getLastScore() {
        return lastScore;
    }

    public int getBaseDimension() {
        return baseDimension;
    }

    public int getCount() {
        return count;
    }

    public int getMinimumScores() {
        return minimumScores;
    }

    public int getLastAnomalyTimeStamp(){
        return lastAnomalyTimeStamp;
    }

}
