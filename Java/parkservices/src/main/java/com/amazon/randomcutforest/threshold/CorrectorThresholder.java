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
import com.amazon.randomcutforest.threshold.state.CorrectorThresholderState;

public class CorrectorThresholder extends BasicThresholder {

    // the upper threshold of scores above which points are likely anomalies
    protected double upperThreshold = 2.0;
    // the upper threshold of scores above which points are likely anomalies
    protected double lowerThreshold = 1.0;
    //initial absolute threshold used to determine anomalies before sufficient values are seen
    protected double initialThreshold = 1.5;
    // used to determine the suprise coefficient above which we can call a potential anomaly
    protected double zFactor = 2.5;
    // a conservative upper bound of zFactor, when we are seeing a continuous run of anomalies
    protected double triggerFactor = 3.0;
    // an upper bound of zFactor and triggerFactor beyond which the point is mathematically anomalous
    // is useful in determining grade
    protected double upperZfactor = 5.0;
    // a parameter that determines if the current potential anomaly is describing the same anomaly
    // within the same shingle or across different time points
    protected double ignoreSimilarFactor = 0.3;
    // uses attribution by default; can be useful without attribution as a general featurization
    protected boolean attributionEnabled = true;
    // is the previously seen point a potential anomaly
    protected boolean previousIsPotentialAnomaly;
    // a statistical measure of score difference (ignoring the anomaly -> not an anomaly transition)
    protected Deviation scoreDiff;
    // saved attribution of the last seen anomaly
    protected DiVector lastAnomalyAttribution;
    // flag that determines if we should dedup similar anomalies not in the same shingle, for example an
    // anomaly, with the same pattern is repeated across more than a shingle
    protected boolean ignoreSimilar;
    // fraction of the grade that comes from absolute scores in the long run
    protected double absoluteScoreFraction = 0.5;
    // minimum number of scores at which

    public CorrectorThresholder(double discount, int baseDimension, int shingleSize, boolean attributionEnabled, double absoluteThreshold,int minimumScores, boolean ignoreSimilar){
        super(discount,baseDimension,shingleSize);
        this.ignoreSimilar = ignoreSimilar;
        this.lowerThreshold = absoluteThreshold;
        this.attributionEnabled = attributionEnabled;
        this.minimumScores = minimumScores;
        this.scoreDiff = new Deviation(discount);
    }

    public CorrectorThresholder(CorrectorThresholderState state, Deviation simpleDeviation,Deviation scoreDiff){
        super(simpleDeviation);
        this.scoreDiff = scoreDiff;
        this.lastAnomalyAttribution = (state.getLastAnomalyAttribution() != null)?new DiVector(state.getLastAnomalyAttribution()):null;
        this.ignoreSimilar = state.isIgnoreSimilar();
        this.shingleSize = state.getShingleSize();
        this.attributionEnabled = state.isAttributionEnabled();
        this.previousIsPotentialAnomaly = state.isPreviousIsPotentialAnomaly();
        this.zFactor = state.getZFactor();
        this.ignoreSimilarFactor=state.getIgnoreSimilarFactor();
        this.upperZfactor = state.getUpperZfactor();
        this.triggerFactor = state.getTriggerFactor();
        this.upperThreshold = state.getUpperThreshold();
        this.lowerThreshold = state.getLowerThreshold();
        this.initialThreshold = state.getInitialThreshold();
        this.absoluteScoreFraction = state.getAbsoluteScoreFraction();
        this.inAnomaly = state.isInAnomaly();
        this.discount = state.getDiscount();
        this.count = state.getCount();
        this.baseDimension = state.getBaseDimension();
        this.minimumScores = state.getMinimumScores();
        this.lastAnomalyScore = state.getLastAnomalyScore();
        this.lastScore = state.getLastScore();
        checkArgument(initialThreshold <= upperThreshold, "incorrect setting of threshold");
        checkArgument(lowerThreshold <= initialThreshold, "incorrect setting of threshold");
        checkArgument(zFactor<=triggerFactor, "incorrect factor settings");
        checkArgument(triggerFactor<= upperZfactor, "incorrect factor settings");
    }

    @Override
    public int process(double newScore, int timeStamp){
        return process(newScore,newScore,null,null,timeStamp);
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
                    if (!inAnomaly && trigger(attribution,timeStamp, null)) {
                        answer = getGrade(newScore);
                        lastAnomalyScore = newScore;
                        inAnomaly = true;
                        lastAnomalyAttribution = new DiVector(attribution);
                        lastAnomalyTimeStamp = timeStamp;
                        if (useLastScore()){
                            scoreDiff.update(Math.max(0,Math.min(upperThreshold,newScore) - lastScore));
                        }
                    } else {
                        if (trigger(attribution, timeStamp, idealAttribution) && newScore > idealScore) {
                            answer = getGrade(newScore);
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
                    answer = getGrade(newScore);
                    lastAnomalyScore = newScore;
                    inAnomaly = true;
                    lastAnomalyTimeStamp = timeStamp;
                    if (useLastScore()){
                        scoreDiff.update(Math.min(upperThreshold,newScore) - lastScore);
                    }
                } else {
                    if (newScore > idealScore) {
                        answer = getGrade(newScore);
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



    protected boolean trigger(DiVector candidate, int timeStamp, DiVector ideal) {
        if (lastAnomalyAttribution == null) {
            return true;
        }
        checkArgument(lastAnomalyAttribution.getDimensions() == candidate.getDimensions(), " error in DiVectors");
        int dimensions = candidate.getDimensions();

        int difference = baseDimension * (timeStamp - lastAnomalyTimeStamp);

        if (difference < dimensions) {
            if (ideal == null) {
                double remainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    remainder += candidate.getHighLowSum(i);
                }
                return (remainder * dimensions / difference - simpleDeviation.getMean() > triggerFactor * simpleDeviation.getDeviation());
            } else {
                double differentialRemainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i]) + Math.abs(candidate.high[i] - ideal.high[i]);
                }
                return (differentialRemainder > ignoreSimilarFactor *lastAnomalyScore) &&
                        (differentialRemainder * dimensions / difference - simpleDeviation.getMean() > triggerFactor * simpleDeviation.getDeviation());
            }
        } else {
            if (!ignoreSimilar) {
                return true;
            }
            double sum = 0;
            for (int i=0;i<dimensions;i++){
                sum += Math.abs(lastAnomalyAttribution.high[i] - candidate.high[i]) +
                        Math.abs(lastAnomalyAttribution.low[i] - candidate.low[i]);
            }
            return (sum > ignoreSimilarFactor *lastScore);
        }
    }

    @Override
    public boolean isPotentialAnomaly(double newScore){
        // cannot change any state

        if (newScore < lowerThreshold) {
            return false;
        }

        if (scoreDiff.getCount() < minimumScores){
                return newScore*minimumScores > ( minimumScores - scoreDiff.getCount()) * upperThreshold
                        + scoreDiff.getCount()*initialThreshold;
        } else {
            if (inAnomaly && shingleSize > 1) {
                return (newScore > basicThreshold() - elasticity);
            } else {
                return (newScore > basicThreshold());
            }
        }
    }

    protected boolean useLastScore(){
        return count>0 && !previousIsPotentialAnomaly;
    }

    @Override
    public double basicThreshold(){
        return simpleDeviation.getMean() + zFactor *(scoreDiff.getDeviation());
    }

    public int getGrade(double score){
        int valuesSeen = Math.min(scoreDiff.getCount(),minimumScores);
        double normalizedScore = Math.min(upperThreshold,score);
        double interpolate = (1-absoluteScoreFraction)*valuesSeen/minimumScores;
        double absolutePart = (1 - interpolate)*( normalizedScore - lowerThreshold)/(upperThreshold -lowerThreshold);
        double factorPart = 0;
        if (valuesSeen > 0) {
            double factor = Math.min((score - simpleDeviation.getMean()) / scoreDiff.getDeviation(), upperZfactor);
            factorPart = interpolate * (1 - 1 / factor) / (1 - 1 / upperZfactor);
        }
        return Math.min(100,(int) Math.floor(100*(absolutePart+factorPart)));
    }

    public boolean isAttributionEnabled() {
        return attributionEnabled;
    }

    public boolean isPreviousIsPotentialAnomaly() {
        return previousIsPotentialAnomaly;
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

    public double getTriggerFactor() {
        return triggerFactor;
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

    public void setTriggerFactor(double factor) {
        triggerFactor = factor;
    }

    public void setzFactor(double factor) {
        zFactor =factor;
    }

    public double getIgnoreSimilarFactor() {
        return ignoreSimilarFactor;
    }

    public void setIgnoreSimilarFactor(double factor){
        ignoreSimilarFactor = factor;
    }

    public void setMoreInformation(boolean flag){
        moreInformation = false;
    }

    public double getAbsoluteScoreFraction() {
        return absoluteScoreFraction;
    }

    public double getInitialThreshold(){
        return initialThreshold;
    }
}
