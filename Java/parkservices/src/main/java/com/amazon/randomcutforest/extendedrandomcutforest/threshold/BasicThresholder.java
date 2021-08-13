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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold;


import com.amazon.randomcutforest.extendedrandomcutforest.threshold.state.BasicThresholderState;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class BasicThresholder implements IThresholder{

    public static int IS_ANOMALY = 1;

    public static int NOT_ANOMALY = 0;

    protected double elasticity = 0.01;

    protected int count = 0;

    protected double primaryDiscount;

    protected double secondaryDiscount;

    // horizon = 0 is short term, switches to secondary
    // horizon = 1 long term, switches to primary
    protected double horizon = 1;

    protected int minimumScores = 10;

    protected Deviation primaryDeviation;

    protected Deviation secondaryDeviation;

    // fraction of the grade that comes from absolute scores in the long run
    protected double absoluteScoreFraction = 0.5;

    // the upper threshold of scores above which points are likely anomalies
    protected double upperThreshold = 2.0;
    // the upper threshold of scores above which points are likely anomalies
    protected double lowerThreshold = 1.0;
    //initial absolute threshold used to determine anomalies before sufficient values are seen
    protected double initialThreshold = 1.5;
    // used to determine the suprise coefficient above which we can call a potential anomaly
    protected double zFactor = 2.5;
    // an upper bound of zFactor and triggerFactor beyond which the point is mathematically anomalous
    // is useful in determining grade
    protected double upperZfactor = 5.0;

    protected double lastScore;


    public BasicThresholder(double discount){
        this.primaryDiscount=discount;
        primaryDeviation = new Deviation(discount);
        this.secondaryDiscount=discount;
        secondaryDeviation = new Deviation(discount);
    }

    public BasicThresholder(Deviation deviation){
        primaryDiscount = deviation.getDiscount();
        secondaryDiscount = primaryDiscount;
        this.primaryDeviation = deviation;
        this.secondaryDeviation = new Deviation(deviation.getDiscount());
    }

    public BasicThresholder(BasicThresholderState state, Deviation primary, Deviation secondary){
        this.primaryDeviation = primary;
        this.secondaryDeviation = secondary;
        this.elasticity =state.getElasticity();
        this.count = state.getCount();
        this.primaryDiscount = state.getPrimaryDiscount();
        this.secondaryDiscount = state.getSecondaryDiscount();
        this.horizon = state.getHorizon();
        this.lastScore = state.getLastScore();
        this.minimumScores = state.getMinimumScores();
        this.absoluteScoreFraction = state.getAbsoluteScoreFraction();
        this.upperThreshold = state.getUpperThreshold();
        this.initialThreshold = state.getInitialThreshold();
        this.lowerThreshold = state.getLowerThreshold();
        this.zFactor = state.getZFactor();
        this.upperZfactor = state.getUpperZfactor();
    }


    protected boolean isDeviationReady(){
        if (count < minimumScores) {
            return false;
        }

        if (horizon == 0) {
            return secondaryDeviation.count >= minimumScores;
        } else if (horizon == 1) {
            return primaryDeviation.count >= minimumScores;
        } else {
            return secondaryDeviation.count >= minimumScores && primaryDeviation.count >= minimumScores;
        }
    }

    protected double intermediateTermFraction(){
        if (count < minimumScores){
            return 0;
        } else if (count> 2*minimumScores){
            return 1;
        } else {
            return (count - minimumScores)*1.0/minimumScores;
        }
    }

    protected boolean isLongTermReady(){
        return (intermediateTermFraction() == 1);
    }

    protected double basicThreshold(double factor) {
        if (!isDeviationReady()) { // count < minimumScore is this branch
            return Math.max(initialThreshold,lowerThreshold);
        } else if (isLongTermReady()) {
            return longTermThreshold(factor);
        } else {
            return Math.max(lowerThreshold,intermediateTermFraction()*longTermThreshold(factor)
                    + (1-intermediateTermFraction())*initialThreshold);
        }

    }
    protected double longTermThreshold(double factor){
        return Math.max(lowerThreshold,primaryDeviation.getMean() + factor * longTermDeviation());
    }

    protected double longTermDeviation(){
        return ( horizon * primaryDeviation.getDeviation() +
                (1-horizon) *secondaryDeviation.getDeviation());
    }


    public double getAnomalyGrade(double score,double factor){
        checkArgument(factor >= zFactor, "incorrect call");
        // please change here is a first cut
        if (isLongTermReady()) {
            if (score < longTermThreshold(factor)) {
                return 0;
            }
            double tFactor = Math.min(upperZfactor, (score - primaryDeviation.getMean()) / longTermDeviation());
            checkArgument(tFactor >= zFactor, "should not be here");
            return (tFactor - zFactor) / (upperZfactor - zFactor);
        } else {
            if (score <basicThreshold(factor)){
                return 0;
            }
            double upper = Math.max(upperThreshold,2*basicThreshold(factor));
            double quasiScore = Math.min(score, upper);
            return (quasiScore - basicThreshold(factor))/(upper - basicThreshold(factor));
        }
    }

    public double getAnomalyGrade(double score){
        return getAnomalyGrade(score,zFactor);
    }

    public double getConfidenceScore(double score){
        // please change
        return 0;
    }

    public void update(double score){

        primaryDeviation.update(score);
        ++count;
    }

    public void update(double primary,double secondary){
        primaryDeviation.update(primary);
        secondaryDeviation.update(secondary);
        ++count;
    }


    public Deviation getPrimaryDeviation() {
        return primaryDeviation;
    }

    public Deviation getSecondaryDeviation() {
        return secondaryDeviation;
    }

    public double getPrimaryDiscount() {
        return primaryDiscount;
    }

    public double getSecondaryDiscount() {
        return secondaryDiscount;
    }

    public double getElasticity() {
        return elasticity;
    }


    public int getCount() {
        return count;
    }

    public int getMinimumScores() {
        return minimumScores;
    }

    public void setElasticity(double elasticity){
        this.elasticity = elasticity;
    }


    public double getzFactor() {
        return zFactor;
    }

    public double getUpperZFactor() {
        return upperZfactor;
    }

    public double getUpperThreshold() {
        return upperThreshold;
    }

    public double getLowerThreshold() {
        return lowerThreshold;
    }

    public void setUpperThreshold(double score) {
        upperThreshold = score;
    }

    public void setzFactor(double factor) {
        zFactor =factor;
    }

    public double getHorizon() {
        return horizon;
    }

    public double getInitialThreshold() {
        return initialThreshold;
    }

    public double getAbsoluteScoreFraction() {
        return absoluteScoreFraction;
    }

    public double getLastScore() {
        return lastScore;
    }

    public void setLowerThreshold(double lowerThreshold) {
        this.lowerThreshold = lowerThreshold;
    }
}
