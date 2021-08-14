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

package com.amazon.randomcutforest.parkservices.threshold;


import lombok.Getter;
import lombok.Setter;

import java.util.List;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

@Getter
@Setter
public class BasicThresholder implements IThresholder {

    // a parameter to make high score regions a contiguous region as opposed to
    // collection of points in and out of the region
    protected double elasticity = 0.01;

    // keeping a count of the values seen because both variables may not be used
    protected int count = 0;

    // horizon = 0 is short term, switches to secondary
    // horizon = 1 long term, switches to primary
    protected double horizon = 0.5;

    // below these many observations, deviation is not useful
    protected int minimumScores = 10;

    protected Deviation primaryDeviation;

    protected Deviation secondaryDeviation;

    // fraction of the grade that comes from absolute scores in the long run
    protected double absoluteScoreFraction = 0.5;

    // the upper threshold of scores above which points are likely anomalies
    protected double upperThreshold = 2.0;
    // the upper threshold of scores above which points are likely anomalies
    protected double lowerThreshold = 1.0;
    // initial absolute threshold used to determine anomalies before sufficient
    // values are seen
    protected double initialThreshold = 1.5;
    // used to determine the suprise coefficient above which we can call a potential
    // anomaly
    protected double zFactor = 2.5;
    // an upper bound of zFactor and triggerFactor beyond which the point is
    // mathematically anomalous
    // is useful in determining grade
    protected double upperZfactor = 5.0;

    public BasicThresholder(double discount) {
        primaryDeviation = new Deviation(discount);
        secondaryDeviation = new Deviation(discount);
    }

    public BasicThresholder(Deviation primary, Deviation secondary) {
        this.primaryDeviation = primary;
        this.secondaryDeviation = secondary;
    }

    /**
     * The constructor creates a thresholder from a sample of scores and a future discount rate
     * @param scores list of scores
     * @param futureAnomalyRate discount/decay factor of scores going forward
     */
    public BasicThresholder(List<Double> scores,double futureAnomalyRate){
        this.primaryDeviation = new Deviation(0);
        this.secondaryDeviation = new Deviation(0);
        scores.forEach(s -> {update(s,s);});
        primaryDeviation.setDiscount(1 - futureAnomalyRate);
    }

    protected boolean isDeviationReady() {
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

    protected double intermediateTermFraction() {
        if (count < minimumScores) {
            return 0;
        } else if (count > 2 * minimumScores) {
            return 1;
        } else {
            return (count - minimumScores) * 1.0 / minimumScores;
        }
    }

    protected boolean isLongTermReady() {
        return (intermediateTermFraction() == 1);
    }

    protected double basicThreshold(double factor) {
        if (!isDeviationReady()) { // count < minimumScore is this branch
            return Math.max(initialThreshold, lowerThreshold);
        } else if (isLongTermReady()) {
            return longTermThreshold(factor);
        } else {
            return Math.max(lowerThreshold, intermediateTermFraction() * longTermThreshold(factor)
                    + (1 - intermediateTermFraction()) * initialThreshold);
        }

    }

    protected double longTermThreshold(double factor) {
        return Math.max(lowerThreshold, primaryDeviation.getMean() + factor * longTermDeviation());
    }

    protected double longTermDeviation() {
        return (horizon * primaryDeviation.getDeviation() + (1 - horizon) * secondaryDeviation.getDeviation());
    }

    public double getAnomalyGrade(double score, double factor) {
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
            if (score < basicThreshold(factor)) {
                return 0;
            }
            double upper = Math.max(upperThreshold, 2 * basicThreshold(factor));
            double quasiScore = Math.min(score, upper);
            return (quasiScore - basicThreshold(factor)) / (upper - basicThreshold(factor));
        }
    }

    public double getAnomalyGrade(double score) {
        return getAnomalyGrade(score, zFactor);
    }

    public double getConfidenceScore(double score) {
        // please change
        return 0;
    }

    public void update(double score) {

        primaryDeviation.update(score);
        ++count;
    }

    public void update(double primary, double secondary) {
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

}
