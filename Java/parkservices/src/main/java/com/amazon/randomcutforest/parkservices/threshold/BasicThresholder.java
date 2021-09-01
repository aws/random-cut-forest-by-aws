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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.List;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class BasicThresholder {

    public static double DEFAULT_ELASTICITY = 0.01;
    public static double DEFAULT_HORIZON = 0.5;
    public static double DEFAULT_HORIZON_ONED = 0.75;
    public static int DEFAULT_MINIMUM_SCORES = 10;
    public static double DEFAULT_ABSOLUTE_SCORE_FRACTION = 0.5;
    public static double DEFAULT_UPPER_THRESHOLD = 2.0;
    public static double DEFAULT_LOWER_THRESHOLD = 1.0;
    public static double DEFAULT_LOWER_THRESHOLD_ONED = 1.1;
    public static double DEFAULT_INITIAL_THRESHOLD = 1.5;
    public static double DEFAULT_Z_FACTOR = 2.5;
    public static double DEFAULT_UPPER_FACTOR = 5.0;

    // a parameter to make high score regions a contiguous region as opposed to
    // collection of points in and out of the region for example the scores
    // can be 1.0, 0.99, 1.0, 0.99 and the thresholds which depend on all scores
    // seen
    // before can be 0.99, 1.0, 0.99, 1.0 .. this parameter smooths the comparison
    protected double elasticity = DEFAULT_ELASTICITY;

    // keeping a count of the values seen because both deviation variables
    // primaryDeviation
    // and secondaryDeviation may not be used always -- in current code, the are
    // both used always
    protected int count = 0;

    // horizon = 0 is short term, switches to secondary
    // horizon = 1 long term, switches to primary
    protected double horizon = DEFAULT_HORIZON;

    // below these many observations, deviation is not useful
    protected int minimumScores = DEFAULT_MINIMUM_SCORES;

    protected Deviation primaryDeviation;

    protected Deviation secondaryDeviation;

    // fraction of the grade that comes from absolute scores in the long run
    protected double absoluteScoreFraction = DEFAULT_ABSOLUTE_SCORE_FRACTION;

    // the upper threshold of scores above which points are anomalies
    protected double upperThreshold = DEFAULT_UPPER_THRESHOLD;
    // the upper threshold of scores above which points are likely anomalies
    protected double lowerThreshold = DEFAULT_LOWER_THRESHOLD;
    // initial absolute threshold used to determine anomalies before sufficient
    // values are seen
    protected double initialThreshold = DEFAULT_INITIAL_THRESHOLD;
    // used to determine the surprise coefficient above which we can call a
    // potential
    // anomaly
    protected double zFactor = DEFAULT_Z_FACTOR;
    // an upper bound of zFactor and triggerFactor beyond which the point is
    // mathematically anomalous
    // is useful in determining grade
    protected double upperZfactor = DEFAULT_UPPER_FACTOR;

    public BasicThresholder(double discount) {
        primaryDeviation = new Deviation(discount);
        secondaryDeviation = new Deviation(discount);
    }

    public BasicThresholder(Deviation primary, Deviation secondary) {
        this.primaryDeviation = primary;
        this.secondaryDeviation = secondary;
    }

    /**
     * The constructor creates a thresholder from a sample of scores and a future
     * discount rate
     * 
     * @param scores            list of scores
     * @param futureAnomalyRate discount/decay factor of scores going forward
     */
    public BasicThresholder(List<Double> scores, double futureAnomalyRate) {
        this.primaryDeviation = new Deviation(0);
        this.secondaryDeviation = new Deviation(0);
        scores.forEach(s -> update(s, s));
        primaryDeviation.setDiscount(futureAnomalyRate);
        secondaryDeviation.setDiscount(futureAnomalyRate);
    }

    public boolean isDeviationReady() {
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
        double a = primaryDeviation.getDeviation();
        double b = secondaryDeviation.getDeviation();
        // the following is a convex combination that allows control of the behavior
        return (horizon * a + (1 - horizon) * b);
    }

    public double getAnomalyGrade(double score, boolean previous, double factor) {
        checkArgument(factor >= zFactor, "incorrect call");
        // please change here is a first cut

        double elasticScore = (previous) ? elasticity : 0;
        if (isLongTermReady()) {
            if (score < longTermThreshold(factor) - elasticScore) {
                return 0;
            }
            double tFactor = upperZfactor;
            if (longTermDeviation() > 0) {
                tFactor = Math.min(tFactor, (score - primaryDeviation.getMean()) / longTermDeviation());
            }

            return (tFactor - zFactor) / (upperZfactor - zFactor);
        } else {
            double t = basicThreshold(factor);
            if (score < t - elasticScore) {
                return 0;
            }
            double upper = Math.max(upperThreshold, 2 * t);
            double quasiScore = Math.min(score, upper);
            return (quasiScore - t) / (upper - t);
        }
    }

    public double getAnomalyGrade(double score, boolean previous) {
        return getAnomalyGrade(score, previous, zFactor);
    }

    protected void updatePrimary(double score) {
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

    public void setZfactor(double factor) {
        zFactor = Math.max(factor, DEFAULT_Z_FACTOR);
        upperZfactor = Math.max(upperZfactor, 2 * zFactor);
    }

    public void setUpperZfactor(double factor) {
        upperZfactor = Math.max(factor, 2 * zFactor);
    }

    public void setLowerThreshold(double lower) {
        lowerThreshold = lower;
        initialThreshold = Math.max(initialThreshold, lowerThreshold);
        upperThreshold = Math.max(upperThreshold, 2 * lowerThreshold);
    }

    public void setInitialThreshold(double initial) {
        initialThreshold = Math.max(initial, lowerThreshold);
        upperThreshold = Math.max(upperThreshold, initial);
    }

    public void setUpperThreshold(double upper) {
        upperThreshold = Math.max(upper, initialThreshold);
        upperThreshold = Math.max(upperThreshold, 2 * lowerThreshold);
    }

    public void setHorizon(double horizon) {
        checkArgument(horizon >= 0 && horizon <= 1, "incorrect horizon parameter");
        this.horizon = horizon;
    }

}
