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


public class BasicThresholder {

    public static int IS_ANOMALY = 1;

    public static int NOT_ANOMALY = 0;

    public static int MORE_INFORMATION = -1;

    protected boolean moreInformation;

    protected boolean inAnomaly = false;

    protected double elasticity = 0.01;

    protected int count = 0;

    protected double discount;

    protected int baseDimension;

    protected int shingleSize;

    protected int minimumScores = 10;

    protected Deviation simpleDeviation;

    protected int lastAnomalyTimeStamp;

    protected double lowerThreshold = 0.8;

    protected double lastAnomalyScore;

    protected double lastScore;

    protected double BASIC_FACTOR = 3.0;

    public BasicThresholder(double discount, int baseDimension, int shingleSize){
        this.discount=discount;
        simpleDeviation = new Deviation(discount);
        this.baseDimension=baseDimension;
        this.shingleSize = shingleSize;
    }

    public BasicThresholder(Deviation deviation){
        this.simpleDeviation = deviation;
        moreInformation = false;
    }

    public double basicThreshold(){
        return simpleDeviation.getMean() + 3 * simpleDeviation.getDeviation();
    }

    public boolean isPotentialAnomaly(double newScore){
        // cannot change any state

        if (count <= minimumScores || newScore < lowerThreshold) {
            return false;
        }

        if (inAnomaly && shingleSize>1) {
            return (newScore > basicThreshold() - elasticity);
        } else {
            return (newScore > basicThreshold());
        }
    }

    public int process(double newScore, int timeStamp){
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

    public double getLowerThreshold() {
        return lowerThreshold;
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

    public int getShingleSize() {
        return shingleSize;
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

    public double getBASIC_FACTOR() {
        return BASIC_FACTOR;
    }

    public void setBasic_Factor(double factor){
        BASIC_FACTOR = factor;
    }

    public void setElasticity(double elasticity){
        this.elasticity = elasticity;
    }

}
