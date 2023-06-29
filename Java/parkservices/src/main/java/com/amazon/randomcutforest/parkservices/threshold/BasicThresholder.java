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
import static java.lang.Math.sqrt;

import java.util.List;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.util.Weighted;

public class BasicThresholder {

    public static double DEFAULT_SCORE_DIFFERENCING = 0.5;
    public static int DEFAULT_MINIMUM_SCORES = 10;
    public static double DEFAULT_FACTOR_ADJUSTMENT_THRESHOLD = 0.9;
    public static double DEFAULT_ABSOLUTE_THRESHOLD = 0.8;
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
    protected double scoreDifferencing = DEFAULT_SCORE_DIFFERENCING;

    // below these many observations, deviation is not useful
    protected int minimumScores = DEFAULT_MINIMUM_SCORES;

    protected Deviation primaryDeviation;

    protected Deviation secondaryDeviation;

    protected Deviation thresholdDeviation;

    protected boolean autoThreshold = DEFAULT_AUTO_THRESHOLD;

    // an absoluteThreshold
    protected double absoluteThreshold = DEFAULT_ABSOLUTE_THRESHOLD;

    // the upper threshold of scores above which points are likely anomalies
    protected double factorAdjustmentThreshold = DEFAULT_FACTOR_ADJUSTMENT_THRESHOLD;
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

    public BasicThresholder(List<Double> scores, double rate) {
        this.primaryDeviation = new Deviation(0);
        this.secondaryDeviation = new Deviation(0);
        this.thresholdDeviation = new Deviation(0);
        if (scores != null) {
            scores.forEach(s -> update(s, s));
        }
        primaryDeviation.setDiscount(rate);
        secondaryDeviation.setDiscount(rate);
        thresholdDeviation.setDiscount(0.1 * rate);
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

        if (scoreDifferencing == 1) {
            return secondaryDeviation.getCount() >= minimumScores;
        } else if (scoreDifferencing == 0) {
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

    @Deprecated
    public double threshold() {
        return getPrimaryThreshold();
    }

    public double getPrimaryThreshold() {
        if (!isDeviationReady()) {
            return 0;
        }
        return max(absoluteThreshold, primaryDeviation.getMean() + zFactor * primaryDeviation.getDeviation());
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
        if (!isDeviationReady()) {
            return 0;
        }
        double tFactor = 2 * zFactor;
        double deviation = primaryDeviation.getDeviation();
        if (deviation > 0) {
            tFactor = min(tFactor, (score - primaryDeviation.getMean()) / deviation);
        }
        double t = (tFactor - zFactor) / (zFactor);
        return max(0, t);
    }

    public Weighted<Double> getPrimaryThresholdAndGrade(double score) {
        if (!isDeviationReady() || score <= 0) {
            return new Weighted<Double>(0.0, 0.0f);
        }
        double threshold = getPrimaryThreshold();
        float grade = (threshold > 0 && score > threshold) ? (float) getPrimaryGrade(score) : 0f;
        return new Weighted<>(threshold, grade);
    }

    @Deprecated
    public double getAnomalyGrade(double score, boolean flag) {
        return getPrimaryGrade(score);
    }

    /**
     * The following adapts the notion of x-sigma (standard deviation) to admit the
     * case that RCF scores are asymmetric and values lower than 1 (closer to 0.5)
     * can be more common; whereas anomalies are typically larger the x-factor is
     * automatically scaled to be calibrated with the average score (bounded below
     * by an absolute constant like 0.7)
     * 
     * @param factor    the factor being scaled
     * @param method    transformation method
     * @param dimension the dimension of the problem (currently unused)
     * @return a scaled value of the factor
     */

    protected double adjustedFactor(double factor, TransformMethod method, int dimension) {
        double correctedFactor = factor;
        double base = primaryDeviation.getMean();
        if (autoThreshold && base < factorAdjustmentThreshold && method != TransformMethod.NONE) {
            correctedFactor = primaryDeviation.getMean() * factor / factorAdjustmentThreshold;
        }
        return max(correctedFactor, MINIMUM_Z_FACTOR);
    }

    /**
     * The following computes the standard deviation of the scores. But we have
     * multiple ways of measuring that -- if the scores are typically symmetric then
     * many of these measures concide. However transformation of the values may
     * cause the score distribution to be unusual. For example, if NORMALIZATION is
     * used then the scores (below the average) end up being close to the average
     * (an example of the asymmetry) and thus only standard deviation is used. But
     * for other distributions we could directly estimate the deviation of the
     * scores below the dynamic mean in an online manner, and we do so in
     * thresholdDeviation. An orthogonal component is the effect of
     * shingling/differencing which connect up the scores from consecutive input.
     * 
     * @param method      transformation method
     * @param shingleSize shinglesize used
     * @return an estimate of long term deviation from mean of a stochastic series
     */
    protected double longTermDeviation(TransformMethod method, int shingleSize) {

        if (shingleSize == 1
                && !(method == TransformMethod.DIFFERENCE || method == TransformMethod.NORMALIZE_DIFFERENCE)) {
            // control the effect of large values above a threshold from raising the
            // threshold
            return min(sqrt(2.0) * thresholdDeviation.getDeviation(), primaryDeviation.getDeviation());
        } else {
            double first = primaryDeviation.getDeviation();
            if (method != TransformMethod.NORMALIZE) {
                first = min(first, sqrt(2.0) * thresholdDeviation.getDeviation());

            }
            // there is a role of differenceing; either by shingling or by explicit
            // transformation
            return scoreDifferencing * first + (1 - scoreDifferencing) * secondaryDeviation.getDeviation();
        }

    }

    public Weighted<Double> getThresholdAndGrade(double score, TransformMethod method, int dimension, int shingleSize) {
        return getThresholdAndGrade(score, zFactor, method, dimension, shingleSize);
    }

    public Weighted<Double> getThresholdAndGrade(double score, double factor, TransformMethod method, int dimension,
            int shingleSize) {
        double intermediateFraction = intermediateTermFraction();
        double newFactor = adjustedFactor(factor, method, dimension);
        double longTerm = longTermDeviation(method, shingleSize);
        double scaledDeviation = (newFactor - 1) * longTerm + primaryDeviation.getDeviation();

        double threshold = (!isDeviationReady()) ? max(initialThreshold, absoluteThreshold)
                : max(absoluteThreshold, intermediateFraction * (primaryDeviation.getMean() + scaledDeviation)
                        + (1 - intermediateFraction) * initialThreshold);
        if (score < threshold || threshold <= 0) {
            return new Weighted<>(threshold, 0);
        } else {
            double base = min(threshold, max(absoluteThreshold, primaryDeviation.getMean()));
            // the value below should not be 0 because of min()
            return new Weighted<>(threshold, getSurpriseIndex(score, base, newFactor, scaledDeviation / newFactor));
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
        if (isDeviationReady()) {
            double tFactor = 2 * factor;
            if (deviation > 0) {
                tFactor = min(factor, (score - base) / deviation);
            }
            return max(0, (float) (tFactor / factor));
        } else {
            return (float) min(1, max(0, (score - absoluteThreshold) / absoluteThreshold));
        }
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

    public void setZfactor(double factor) {
        zFactor = factor;
    }

    /**
     * sets the lower threshold -- which is used to scale the factor variable
     */
    public void setLowerThreshold(double lower) {
        factorAdjustmentThreshold = lower;
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

    public void setScoreDifferencing(double scoreDifferencing) {
        checkArgument(scoreDifferencing >= 0 && scoreDifferencing <= 1, "incorrect score differencing parameter");
        this.scoreDifferencing = scoreDifferencing;
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

    public double getLowerThreshold() {
        return factorAdjustmentThreshold;
    }

    public double getInitialThreshold() {
        return initialThreshold;
    }

    public double getScoreDifferencing() {
        return scoreDifferencing;
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
