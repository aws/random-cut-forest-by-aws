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
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.List;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.util.Weighted;

public class BasicThresholder {

    public static double DEFAULT_THRESHOLD_PERSISTENCE = 0.5;
    public static int DEFAULT_MINIMUM_SCORES = 10;
    public static double DEFAULT_LOWER_THRESHOLD = 0.9;
    public static double DEFAULT_LOWER_THRESHOLD_NORMALIZED = 0.8;
    public static double DEFAULT_ABSOLUTE_THRESHOLD = 0.6;
    public static double DEFAULT_INITIAL_THRESHOLD = 1.5;
    public static double DEFAULT_Z_FACTOR = 3.0;
    public static double MINIMUM_Z_FACTOR = 2.0;
    public static boolean DEFAULT_AUTO_THRESHOLD = true;
    public static int DEFAULT_DEVIATION_STATES = 3;

    // keeping a count of the values seen because both deviation variables
    // primaryDeviation
    // and secondaryDeviation may not be used always
    protected int count = 0;

    // horizon = 0 is short term, switches to secondary
    // horizon = 1 long term, switches to primary
    protected double thresholdPersistence = DEFAULT_THRESHOLD_PERSISTENCE;

    // below these many observations, deviation is not useful
    protected int minimumScores = DEFAULT_MINIMUM_SCORES;

    protected Deviation primaryDeviation;

    protected Deviation secondaryDeviation;

    protected Deviation thresholdDeviation;

    protected boolean autoThreshold = DEFAULT_AUTO_THRESHOLD;

    // an absoluteThreshold
    protected double absoluteThreshold = DEFAULT_ABSOLUTE_THRESHOLD;

    // the upper threshold of scores above which points are likely anomalies
    protected double lowerThreshold = DEFAULT_LOWER_THRESHOLD;
    // initial absolute threshold used to determine anomalies before sufficient
    // values are seen
    protected double initialThreshold = DEFAULT_INITIAL_THRESHOLD;
    // used to determine the surprise coefficient above which we can call a
    // potential anomaly
    protected double zFactor = DEFAULT_Z_FACTOR;

    public BasicThresholder(double primaryDiscount, double secondaryDiscount, boolean adjust) {
        primaryDeviation = new Deviation(primaryDiscount);
        secondaryDeviation = new Deviation(secondaryDiscount);
        // a longer horizon to adjust
        thresholdDeviation = new Deviation(primaryDiscount / 2);
        autoThreshold = adjust;
    }

    public BasicThresholder(double discount) {
        this(discount, discount, false);
    }

    public BasicThresholder(Deviation[] deviations) {
        if (deviations == null || deviations.length != DEFAULT_DEVIATION_STATES) {
            double timeDecay = 1.0 / (DEFAULT_SAMPLE_SIZE * DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY);
            this.primaryDeviation = new Deviation(timeDecay);
            this.secondaryDeviation = new Deviation(timeDecay);
            this.thresholdDeviation = new Deviation(0.1 * timeDecay);
        } else {
            this.primaryDeviation = deviations[0];
            this.secondaryDeviation = deviations[1];
            this.thresholdDeviation = deviations[2];
        }
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
        this.thresholdDeviation = new Deviation(0);
        if (scores != null) {
            scores.forEach(s -> update(s, s));
        }
        primaryDeviation.setDiscount(futureAnomalyRate);
        secondaryDeviation.setDiscount(futureAnomalyRate);
        thresholdDeviation.setDiscount(futureAnomalyRate / 2);
    }

    /**
     * a boolean that determines if enough values have been seen to be able to
     * discern deviations
     * 
     * @return true/false based on counts of various statistic
     */
    public boolean isDeviationReady() {
        if (count < minimumScores) {
            return false;
        }

        if (thresholdPersistence == 0) {
            return secondaryDeviation.getCount() >= minimumScores;
        } else if (thresholdPersistence == 1) {
            return primaryDeviation.getCount() >= minimumScores;
        } else {
            return secondaryDeviation.getCount() >= minimumScores && primaryDeviation.getCount() >= minimumScores;
        }
    }

    /**
     * this function helps switch from short term (not able to use deviation, using
     * absolute scores) which is the first minimumScores observations of the scoring
     * function to using deviation (and not using absokute scores, except as a lower
     * bound) at 2*minimumScores It is often the case that the data has "run"
     * effects and the initial scopres can all come in low or can all come in high
     * 
     * @return a parameter that helps smoot transition of initial to long term
     *         behavior
     */
    protected double intermediateTermFraction() {
        if (count < minimumScores) {
            return 0;
        } else if (count > 2 * minimumScores) {
            return 1;
        } else {
            return (count - minimumScores) * 1.0 / minimumScores;
        }
    }

    protected double threshold(double factor, double intermediateTermFraction, TransformMethod method, int dimension) {
        if (!isDeviationReady()) { // count < minimumScore is this branch
            return max(initialThreshold, absoluteThreshold);
        } else {
            return max(absoluteThreshold,
                    intermediateTermFraction
                            * (primaryDeviation.getMean()
                                    + adjustedFactor(factor, method, dimension) * longTermDeviation(method, dimension))
                            + (1 - intermediateTermFraction) * initialThreshold);
        }

    }

    protected double adjustedFactor(double factor, TransformMethod method, int dimension) {
        double correctedFactor = factor;
        double base = primaryDeviation.getMean();
        if (autoThreshold && base < lowerThreshold && method != TransformMethod.NORMALIZE) {
            correctedFactor = primaryDeviation.getMean() * factor / lowerThreshold;
        }
        return max(correctedFactor, MINIMUM_Z_FACTOR);
    }

    protected double longTermDeviation(TransformMethod method, int dimension) {
        double a = primaryDeviation.getDeviation();
        double b = secondaryDeviation.getDeviation();
        if (dimension > 1 && (method == TransformMethod.NORMALIZE_DIFFERENCE || method == TransformMethod.DIFFERENCE)) {
            a = (a + thresholdDeviation.getDeviation()) / 2;
        }
        // the following is a convex combination that allows control of the behavior
        return (thresholdPersistence * a + (1 - thresholdPersistence) * b);
    }

    @Deprecated
    public double threshold() {
        return getPrimaryThreshold();
    }

    public double getPrimaryThreshold() {
        return primaryDeviation.getMean() + zFactor * primaryDeviation.getDeviation();
    }

    /**
     * The simplest thresholder that does not use any auxilliary correction, an can
     * be used for multiple scoring capabilities.
     * 
     * @param score the value being thresholded
     * @return a computation of grade between [-1,1], grades in the range (0,1] are
     *         to be considered anomalous
     */
    public double getPrimaryGrade(double score) {
        double tFactor = 2 * zFactor;
        double deviation = primaryDeviation.getDeviation();
        if (deviation > 0) {
            tFactor = min(tFactor, (score - primaryDeviation.getMean()) / deviation);
        }
        double t = (tFactor - zFactor) / (zFactor);
        return max(0, t);
    }

    @Deprecated
    public double getAnomalyGrade(double score, boolean flag) {
        return getPrimaryGrade(score);
    }

    public Weighted<Double> getThresholdAndGrade(double score, TransformMethod method, int dimension) {
        return getThresholdAndGrade(score, zFactor, method, dimension);
    }

    public Weighted<Double> getThresholdAndGrade(double score, double factor, TransformMethod method, int dimension) {
        double intermediateFraction = intermediateTermFraction();
        double threshold = threshold(factor, intermediateFraction, method, dimension);
        if (score < threshold) {
            return new Weighted<>(threshold, 0);
        } else {
            double base = min(threshold, primaryDeviation.getMean());
            double newFactor = adjustedFactor(factor, method, dimension);
            double deviation = longTermDeviation(method, dimension);
            // the value below should not be 0 because of min()
            return new Weighted<>(threshold, getSurpriseIndex(score, base, newFactor, deviation));
        }
    }

    /**
     * how surprised are seeing a value from a series with mean base with deviation,
     * where factor controls the separation
     * 
     * @param score     score
     * @param base      mean of series
     * @param factor    control parameter for determining surprise
     * @param deviation relevant deviation for the series
     * @return a clipped value of the "surpise" index
     */
    protected float getSurpriseIndex(double score, double base, double factor, double deviation) {
        double tFactor = 2 * factor;
        if (deviation > 0) {
            tFactor = min(tFactor, (score - base) / deviation);
        }
        return max(0, (float) ((tFactor - factor) / (factor)));
    }

    // mean or below; uses the asymmetry of the RCF score
    protected void updateThreshold(double score) {
        double gap = primaryDeviation.getMean() - score;
        if (gap > 0) {
            thresholdDeviation.update(gap);
        }
    }

    protected void updatePrimary(double score) {
        updateThreshold(score);
        primaryDeviation.update(score);
        ++count;
    }

    public void update(double primary, double secondary) {
        updateThreshold(primary);
        primaryDeviation.update(primary);
        secondaryDeviation.update(secondary);
        ++count;
    }

    public void update(double score, double secondScore, double lastScore, TransformMethod method) {
        update(score, secondScore - lastScore);
    }

    public Deviation getPrimaryDeviation() {
        return primaryDeviation;
    }

    public Deviation getSecondaryDeviation() {
        return secondaryDeviation;
    }

    public Deviation getThresholdDeviation() {
        return thresholdDeviation;
    }

    public void setZfactor(double factor) {
        zFactor = factor;
    }

    /**
     * sets the lower threshold -- which is used to scale the factor variable
     */
    public void setLowerThreshold(double lower) {
        lowerThreshold = lower;
    }

    /**
     * 
     * @param value absolute lower bound thresholds
     */
    public void setAbsoluteThreshold(double value) {
        absoluteThreshold = value;
    }

    public void setInitialThreshold(double initial) {
        initialThreshold = initial;
    }

    public void setThresholdPersistence(double horizon) {
        checkArgument(horizon >= 0 && horizon <= 1, "incorrect threshold horizon parameter");
        this.thresholdPersistence = horizon;
    }

    // to be updated as more deviations are added
    public Deviation[] getDeviations() {
        Deviation[] deviations = new Deviation[DEFAULT_DEVIATION_STATES];
        deviations[0] = primaryDeviation.copy();
        deviations[1] = secondaryDeviation.copy();
        deviations[2] = thresholdDeviation.copy();
        return deviations;
    }

    public boolean isAutoThreshold() {
        return autoThreshold;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public double getAbsoluteThreshold() {
        return absoluteThreshold;
    }

    public double getInitialThreshold() {
        return initialThreshold;
    }

    public double getLowerThreshold() {
        return lowerThreshold;
    }

    public double getThresholdPersistence() {
        return thresholdPersistence;
    }

    public double getZFactor() {
        return zFactor;
    }

    public int getMinimumScores() {
        return minimumScores;
    }

    public void setMinimumScores(int minimumScores) {
        this.minimumScores = minimumScores;
    }

    public void setAutoThreshold(boolean autoThreshold) {
        this.autoThreshold = autoThreshold;
    }
}
