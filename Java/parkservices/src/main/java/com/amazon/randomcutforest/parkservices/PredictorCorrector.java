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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.parkservices.config.CorrectionMode.ALERT_ONCE;
import static com.amazon.randomcutforest.parkservices.config.CorrectionMode.CONDITIONAL_FORECAST;
import static com.amazon.randomcutforest.parkservices.config.CorrectionMode.DATA_DRIFT;
import static com.amazon.randomcutforest.parkservices.config.CorrectionMode.NONE;
import static com.amazon.randomcutforest.preprocessor.Preprocessor.DEFAULT_NORMALIZATION_PRECISION;
import static java.lang.Math.exp;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.config.CorrectionMode;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.statistics.Deviation;
import com.amazon.randomcutforest.util.Weighted;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
public class PredictorCorrector {
    private static double DEFAULT_DIFFERENTIAL_FACTOR = 0.3;

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 5;

    public static double DEFAULT_NOISE_SUPPRESSION_FACTOR = 1.0;

    public static double DEFAULT_MULTI_MODE_SAMPLING_RATE = 0.1;

    public static double DEFAULT_SAMPLING_SUPPORT = 0.1;

    public static int DEFAULT_RUN_ALLOWED = 2;
    // the above will trigger on the 4th occurrence, because the first is not
    // counted in the run

    protected static int NUMBER_OF_MODES = 2;

    protected final static int EXPECTED_INVERSE_DEPTH_INDEX = 0;

    protected final static int DISTANCE_INDEX = 1;

    // the following vectors enable suppression of anomalies
    // the first pair correspond to additive differences
    // the second pair correspond to multiplicative differences
    // which are not meaningful for differenced operations

    double[] ignoreNearExpectedFromBelow;

    double[] ignoreNearExpectedFromAbove;

    double[] ignoreNearExpectedFromBelowByRatio;

    double[] ignoreNearExpectedFromAboveByRatio;

    // for anomaly description we would only look at these many top attributors
    // AExpected value is not well-defined when this number is greater than 1
    // that being said there is no formal restriction other than the fact that the
    // answers would be error prone as this parameter is raised.
    protected int numberOfAttributors = DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS;

    protected double[] lastScore = new double[NUMBER_OF_MODES];

    protected ScoringStrategy lastStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;

    protected BasicThresholder[] thresholders;

    protected int baseDimension;

    protected long randomSeed;

    protected double[] modeInformation;

    protected Deviation[] deviationsActual;

    protected Deviation[] deviationsExpected;

    protected double samplingRate = DEFAULT_MULTI_MODE_SAMPLING_RATE;

    protected double noiseFactor = DEFAULT_NOISE_SUPPRESSION_FACTOR;

    protected boolean autoAdjust = false;

    protected RCFComputeDescriptor lastDescriptor;

    protected int runLength;

    protected boolean ignoreDrift = false;

    protected double samplingSupport = DEFAULT_SAMPLING_SUPPORT;

    public PredictorCorrector(double timeDecay, double anomalyRate, boolean adjust, int baseDimension,
            long randomSeed) {
        this.thresholders = new BasicThresholder[NUMBER_OF_MODES];
        thresholders[0] = new BasicThresholder(timeDecay, anomalyRate, adjust);
        thresholders[1] = new BasicThresholder(timeDecay);
        this.baseDimension = baseDimension;
        this.randomSeed = randomSeed;
        this.autoAdjust = adjust;
        if (adjust) {
            this.deviationsActual = new Deviation[baseDimension];
            this.deviationsExpected = new Deviation[baseDimension];
            for (int i = 0; i < baseDimension; i++) {
                this.deviationsActual[i] = new Deviation(timeDecay);
                this.deviationsExpected[i] = new Deviation(timeDecay);
            }
        }
        ignoreNearExpectedFromAbove = new double[baseDimension];
        ignoreNearExpectedFromBelow = new double[baseDimension];
        ignoreNearExpectedFromAboveByRatio = new double[baseDimension];
        ignoreNearExpectedFromBelowByRatio = new double[baseDimension];
    }

    // for mappers
    public PredictorCorrector(BasicThresholder[] thresholders, Deviation[] deviations, int baseDimension,
            long randomSeed) {
        checkArgument(thresholders.length > 0, " cannot be empty");
        this.thresholders = new BasicThresholder[NUMBER_OF_MODES];
        int size = min(thresholders.length, NUMBER_OF_MODES);
        for (int i = 0; i < size; i++) {
            this.thresholders[i] = thresholders[i];
        }
        Deviation deviation = thresholders[0].getPrimaryDeviation();
        for (int i = size; i < NUMBER_OF_MODES; i++) {
            this.thresholders[i] = new BasicThresholder(thresholders[0].getPrimaryDeviation().getDiscount());
        }
        this.deviationsActual = new Deviation[baseDimension];
        this.deviationsExpected = new Deviation[baseDimension];
        if (deviations != null) {
            checkArgument(deviations.length == 2 * baseDimension, "incorrect state");
            for (int i = 0; i < baseDimension; i++) {
                deviationsActual[i] = deviations[i];
            }
            for (int i = 0; i < baseDimension; i++) {
                deviationsExpected[i] = deviations[i + baseDimension];
            }
        }
        this.baseDimension = baseDimension;
        this.randomSeed = randomSeed;
        ignoreNearExpectedFromAbove = new double[baseDimension];
        ignoreNearExpectedFromBelow = new double[baseDimension];
        ignoreNearExpectedFromAboveByRatio = new double[baseDimension];
        ignoreNearExpectedFromBelowByRatio = new double[baseDimension];
    }

    public PredictorCorrector(BasicThresholder thresholder, int baseDimension) {
        this(new BasicThresholder[] { thresholder }, null, baseDimension, 0L);
    }

    protected double nextDouble() {
        Random random = new Random(randomSeed);
        randomSeed = random.nextLong();
        return random.nextDouble();
    }

    /**
     * uses the attribution information to find the time slice which contributed
     * most to the anomaly note that the basic length of the vectors is shingleSize
     * * basDimension; the startIndex corresponds to the shingle entry beyond which
     * the search is performed. if two anomalies are in a shingle it would focus on
     * later one, the previous one would have been (hopefully) reported earlier.
     *
     * @param diVector      attribution of current shingle
     * @param baseDimension number of attributes/variables in original data
     * @param startIndex    time slice of the farthest in the past we are looking
     * @return the index (in this shingle) which has the largest contributions
     */
    protected int maxContribution(DiVector diVector, int baseDimension, int startIndex) {
        double val = 0;
        int index = startIndex;
        int position = diVector.getDimensions() + startIndex * baseDimension;
        for (int i = 0; i < baseDimension; i++) {
            val += diVector.getHighLowSum(i + position);
        }
        for (int i = position + baseDimension; i < diVector.getDimensions(); i += baseDimension) {
            double sum = 0;
            for (int j = 0; j < baseDimension; j++) {
                sum += diVector.getHighLowSum(i + j);
            }
            if (sum > val) {
                val = sum;
                index = (i - diVector.getDimensions()) / baseDimension;
            }
        }
        return index;
    }

    /**
     * the following creates the expected point based on RCF forecasting
     * 
     * @param diVector      the attribution vector that is used to choose which
     *                      elements are to be predicted
     * @param position      the block of (multivariate) elements we are focusing on
     * @param baseDimension the base dimension of the block
     * @param point         the point near which we wish to predict
     * @param forest        the resident RCF
     * @return a vector that is most likely, conditioned on changing a few elements
     *         in the block at position
     */
    protected float[] getExpectedPoint(DiVector diVector, int position, int baseDimension, float[] point,
            RandomCutForest forest) {
        int[] likelyMissingIndices;
        if (baseDimension == 1) {
            likelyMissingIndices = new int[] { position };
        } else {
            double sum = 0;
            double[] values = new double[baseDimension];
            for (int i = 0; i < baseDimension; i++) {
                sum += values[i] = diVector.getHighLowSum(i + position);
            }
            Arrays.sort(values);
            int pick = 1;
            if (values[baseDimension - pick] < 0.1 * sum) {
                // largest contributor is only 10 percent; there are too many to predict
                return null;
            }

            double threshold = min(0.1 * sum, 0.1);
            while (pick < baseDimension && values[baseDimension - pick - 1] >= threshold) {
                ++pick;
            }

            if (pick > numberOfAttributors) {
                // we chose everything; not usable
                return null;
            }

            double cutoff = values[baseDimension - pick];
            likelyMissingIndices = new int[pick];
            int count = 0;
            for (int i = 0; i < baseDimension && count < pick; i++) {
                if (diVector.getHighLowSum(i + position) >= cutoff
                        && (count == 0 || diVector.getHighLowSum(i + position) > sum * 0.1)) {
                    likelyMissingIndices[count++] = position + i;
                }
            }
        }
        if (likelyMissingIndices.length > 0.5 * forest.getDimensions()) {
            return null;
        } else {
            return forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
        }
    }

    /**
     * a subroutine that helps eliminates flagging anomalies too close to a
     * previously flagged anomaly -- this avoids the repetition due to shingling;
     * but still can detect some anomalies if the deviations are usual
     * 
     * @param candidate             the candidate attribution of the point
     * @param difference            the gap (in RCF space) from the last anomaly
     * @param baseDimension         the size of a block
     * @param ideal                 an idealized version of the candidate (can be
     *                              null) where the most offending elements are
     *                              imputed out
     * @param lastAnomalyDescriptor the description of the last anomaly
     * @param workingThreshold      the threshold to exceed
     * @return true if the candidate is sufficiently different and false otherwise
     */

    protected boolean trigger(DiVector candidate, int difference, int baseDimension, DiVector ideal,
            RCFComputeDescriptor lastAnomalyDescriptor, double workingThreshold) {
        int dimensions = candidate.getDimensions();
        if (difference >= dimensions || ideal == null) {
            return true;
        }
        double lastAnomalyScore = lastAnomalyDescriptor.getRCFScore();
        double differentialRemainder = 0;
        for (int i = dimensions - difference; i < dimensions; i++) {
            differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i])
                    + Math.abs(candidate.high[i] - ideal.high[i]);
        }
        return (differentialRemainder > DEFAULT_DIFFERENTIAL_FACTOR * lastAnomalyScore)
                && differentialRemainder * dimensions / difference > 1.2 * workingThreshold;

    }

    /**
     * corrects the effect of a last anomaly -- note that an anomaly by definition
     * will alter the shift and scale of transformations. This computation fixes one
     * single large anomaly.
     * 
     * @param transformMethod       the transformation method used
     * @param gap                   the number of steps the anomaly occurred in the
     *                              past
     * @param lastAnomalyDescriptor the descriptor of the last anomaly
     * @param currentScale          the current scale
     * @return a correction vector
     */
    public double[] getCorrectionOfLastAnomaly(TransformMethod transformMethod, int gap,
            RCFComputeDescriptor lastAnomalyDescriptor, double[] currentScale) {
        double[] deltaShift = lastAnomalyDescriptor.getDeltaShift();
        double[] answer = new double[currentScale.length];
        // correct the effect of shifts in last observed anomaly because the anomaly may
        // have skewed the shift and scale -- but the gap cannot last forever
        // otherwise this will always change the point and force a costlier path
        if (deltaShift != null && gap < 2 * lastAnomalyDescriptor.getShingleSize()
                && (transformMethod == TransformMethod.NORMALIZE || transformMethod == TransformMethod.SUBTRACT_MA)) {
            double factor = exp(-gap * lastAnomalyDescriptor.getTransformDecay());
            for (int y = 0; y < answer.length; y++) {
                answer[y] = (currentScale[y] == 0) ? 0 : deltaShift[y] * factor / currentScale[y];
            }
        }
        return answer;
    }

    /**
     * a first stage corrector that attempts to fix the after effects of a previous
     * anomaly which may be in the shingle, or just preceding the shingle
     *
     * @param point                 the current (transformed) point under evaluation
     * @param gap                   the relative position of the previous anomaly
     *                              being corrected
     * @param shingleSize           size of the shingle
     * @param baseDimensions        number of dimensions in each shingle
     * @param currentScale          scale for current point
     * @param transformMethod       transformation Method
     * @param lastAnomalyDescriptor description of the last anomaly
     * @return the corrected point
     */
    protected <P extends AnomalyDescriptor> float[] applyPastCorrector(float[] point, int gap, int shingleSize,
            int baseDimensions, double[] currentScale, TransformMethod transformMethod,
            RCFComputeDescriptor lastAnomalyDescriptor) {
        float[] correctedPoint = Arrays.copyOf(point, point.length);

        // following will fail for first 100ish points and if dimension < 3
        if (lastAnomalyDescriptor.getExpectedRCFPoint() != null) {
            float[] lastExpectedPoint = lastAnomalyDescriptor.getExpectedRCFPoint();
            float[] lastAnomalyPoint = lastAnomalyDescriptor.getRCFPoint();
            int lastRelativeIndex = lastAnomalyDescriptor.getRelativeIndex();

            // the following will fail for shingleSize 1
            if (gap < shingleSize) {
                System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                        point.length - gap * baseDimensions);
            }
            if (gap <= shingleSize && lastRelativeIndex == 0) {
                if (transformMethod == TransformMethod.DIFFERENCE
                        || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
                    for (int y = 0; y < baseDimensions; y++) {
                        correctedPoint[point.length - gap * baseDimensions
                                + y] += lastAnomalyPoint[point.length - baseDimensions + y]
                                        - lastExpectedPoint[point.length - baseDimensions + y];
                    }
                }
                if (lastAnomalyDescriptor.getForestMode() == ForestMode.TIME_AUGMENTED) {
                    // definitely correct the time dimension which is always differenced
                    // this applies to the non-differenced cases
                    correctedPoint[point.length - (gap - 1) * baseDimensions - 1] += lastAnomalyPoint[point.length - 1]
                            - lastExpectedPoint[point.length - 1];
                }
            }
        }
        double[] correctionVector = getCorrectionOfLastAnomaly(transformMethod, gap, lastAnomalyDescriptor,
                currentScale);
        int number = min(gap, shingleSize);
        for (int y = 0; y < baseDimensions; y++) {
            for (int j = 0; j < number; j++) {
                correctedPoint[point.length - (number - j) * baseDimensions + y] += correctionVector[y];
            }
        }
        return correctedPoint;
    }

    /**
     * The following verifies that the overall shingled point is not explainable by
     * floating point precision. It then verifies that the point is not within
     * noiseFactor of the standard deviation of the successive differences (in the
     * multivariate setting). Finally, it caps the maximum grade possible for this
     * point
     * 
     * @param result the transcript of the current point
     * @param point  the current point
     * @param <P>    Either AnomalyDescriptor of ForecastDescriptor
     * @return a cap on the grade (can be 0 for filtering out)
     */
    protected <P extends AnomalyDescriptor> double centeredTransformPass(P result, float[] point) {
        double maxFactor = 0;
        // check entire point or some large value
        double[] scale = result.getScale();
        double[] shift = result.getShift();
        double[] deviations = result.getDeviations();
        for (int i = 0; i < point.length && maxFactor == 0; i++) {
            double scaleFactor = (scale == null) ? 1.0 : scale[i % baseDimension];
            double shiftBase = (shift == null) ? 0 : shift[i % baseDimension];
            if (Math.abs(point[i]) * scaleFactor > DEFAULT_NORMALIZATION_PRECISION * (1 + Math.abs(shiftBase))) {
                maxFactor = 1;
            }
        }
        // check most recent input
        if (maxFactor > 0) {
            for (int i = 0; i < baseDimension; i++) {
                double scaleFactor = (scale == null) ? 1.0 : Math.abs(scale[i]);
                double z = Math.abs(point[point.length - baseDimension + i]) * scaleFactor;
                double deviation = (deviations == null) ? 0 : Math.abs(deviations[i + baseDimension]);
                if (z > noiseFactor * deviation) {
                    maxFactor = (deviation == 0) ? 1 : min(1.0, max(maxFactor, z / (3 * deviation)));
                }
            }
        }
        return maxFactor;
    }

    /**
     * The following is useful for managing late detection of anomalies -- this
     * calculates the zig-zag over the values in the late detection
     * 
     * @param point         the point being scored
     * @param startPosition the position of the block where we think the anomaly
     *                      started
     * @param index         the specific index in the block being tracked
     * @param baseDimension the size of the block
     * @param differenced   has differencing been performed already
     * @return the average L1 deviation
     */
    double calculatePathDeviation(float[] point, int startPosition, int index, int baseDimension, boolean differenced) {
        int position = startPosition;
        double variation = 0;
        int observation = 0;
        while (position + index + baseDimension < point.length) {
            variation += (differenced) ? Math.abs(point[position + index])
                    : Math.abs(point[position + index] - point[position + baseDimension + index]);
            position += baseDimension;
            ++observation;
        }
        return (observation == 0) ? 0 : variation / observation;
    }

    protected <P extends AnomalyDescriptor> DiVector constructUncertaintyBox(float[] point, int startPosition,
            P result) {
        TransformMethod method = result.getTransformMethod();
        boolean differenced = (method == TransformMethod.DIFFERENCE)
                || (method == TransformMethod.NORMALIZE_DIFFERENCE);
        double[] scale = result.getScale();
        double[] shift = result.getShift();
        int baseDimensions = result.getDimension() / result.getShingleSize();
        double[] gapLow = new double[baseDimensions];
        double[] gapHigh = new double[baseDimensions];
        for (int y = 0; y < baseDimensions; y++) {
            double a = scale[y] * point[startPosition + y];
            double shiftBase = shift[y];
            double shiftAmount = 0;
            if (shiftBase != 0) {
                shiftAmount += DEFAULT_NORMALIZATION_PRECISION * (scale[y] + Math.abs(shiftBase));
            }
            double pathGap = calculatePathDeviation(point, startPosition, y, baseDimension, differenced);
            double noiseGap = noiseFactor * result.getDeviations()[baseDimension + y];
            double gap = max(scale[y] * pathGap, noiseGap) + shiftAmount + DEFAULT_NORMALIZATION_PRECISION;
            gapLow[y] = max(max(ignoreNearExpectedFromBelow[y], ignoreNearExpectedFromBelowByRatio[y] * Math.abs(a)),
                    gap);
            gapHigh[y] = max(max(ignoreNearExpectedFromAbove[y], ignoreNearExpectedFromAboveByRatio[y] * Math.abs(a)),
                    gap);
        }
        return new DiVector(gapHigh, gapLow);
    }

    protected boolean withinGap(DiVector gap, int startPosition, double[] scale, float[] point, float[] otherPoint,
            int baseDimension) {
        boolean answer = false;
        // only for input dimensions, for which scale is defined currently
        for (int y = 0; y < baseDimension && !answer; y++) {
            double a = scale[y] * point[startPosition + y];
            double b = scale[y] * otherPoint[startPosition + y];
            boolean lower = (a < b - gap.low[y]);
            boolean upper = (a > b + gap.high[y]);
            answer = lower || upper;
        }
        return !answer;
    }

    /**
     * uses the native approximate near neighbor in RCF to determine what fraction
     * of samples from different trees are in the uncertainty box around the queried
     * point
     * 
     * @param uncertaintyBox the potentially asymmetric box around a point (original
     *                       space)
     * @param point          the point in question
     * @param correctedPoint any correction applied to the point based on prior
     *                       anomalies
     * @param startPosition  the potential location of the anomaly
     * @param result         the transcript of the current estimation
     * @param forest         the resident RCF
     * @param <P>            an extension of AnomalyDescriptor (to support forecast)
     * @return true if there is enough mass within the box
     */
    protected <P extends AnomalyDescriptor> boolean explainedByConditionalField(DiVector uncertaintyBox, float[] point,
            float[] correctedPoint, int startPosition, P result, RandomCutForest forest) {
        List<Neighbor> list = forest.getNearNeighborsInSample(correctedPoint);
        double averageDistance = list.stream().mapToDouble(e -> e.distance).average().getAsDouble();
        double weight = 0;
        for (Neighbor e : list) {
            if (e.distance < 1.1 * averageDistance && withinGap(uncertaintyBox, startPosition, result.getScale(), point,
                    e.point, point.length / result.getShingleSize())) {
                weight += e.count;
            }
        }
        return (weight >= samplingSupport * forest.getNumberOfTrees());
    }

    /**
     * populates the scores and sets the score and attribution vectors; note some
     * attributions can remain null (for efficiency reasons)
     *
     * @param strategy          the scoring strategy
     * @param point             the current point being evaluated
     * @param forest            the resident RCF
     * @param scoreVector       the vector of scores
     * @param attributionVector the vector of attributions
     * @return the index of the score/attribution that is relevant
     */

    protected int populateScores(ScoringStrategy strategy, float[] point, RandomCutForest forest, double[] scoreVector,
            DiVector[] attributionVector) {
        if (strategy != ScoringStrategy.DISTANCE) {
            scoreVector[EXPECTED_INVERSE_DEPTH_INDEX] = forest.getAnomalyScore(point);
            if (strategy == ScoringStrategy.MULTI_MODE || strategy == ScoringStrategy.MULTI_MODE_RECALL) {
                attributionVector[DISTANCE_INDEX] = forest.getSimpleDensity(point).distances;
                scoreVector[DISTANCE_INDEX] = attributionVector[DISTANCE_INDEX].getHighLowSum();
            }
            return 0;
        } else {
            attributionVector[DISTANCE_INDEX] = forest.getSimpleDensity(point).distances;
            scoreVector[DISTANCE_INDEX] = attributionVector[DISTANCE_INDEX].getHighLowSum();
            return 1;
        }
    }

    /**
     * returned the attribution vector; it tries to reuse cached version to save
     * computation
     * 
     * @param choice            the mode of the attribution in question
     * @param point             the point being considered
     * @param attributionVector the vector (cached) of attributions
     * @param forest            the resident RCF
     * @return the attribution correspond to the mode of attribution
     */
    DiVector getCachedAttribution(int choice, float[] point, DiVector[] attributionVector, RandomCutForest forest) {
        if (attributionVector[choice] == null) {
            checkArgument(choice == EXPECTED_INVERSE_DEPTH_INDEX, "incorrect cached state of scores");
            attributionVector[EXPECTED_INVERSE_DEPTH_INDEX] = forest.getAnomalyAttribution(point);
        }
        return attributionVector[choice];
    }

    /**
     * computes the attribution of a (candidate) point based on mode, when the
     * results are not expected to be cached
     * 
     * @param choice the mode
     * @param point  the point in question
     * @param forest the resident RCF
     * @return the attribution of that mode
     */
    DiVector getNewAttribution(int choice, float[] point, RandomCutForest forest) {
        if (choice == EXPECTED_INVERSE_DEPTH_INDEX) {
            return forest.getAnomalyAttribution(point);
        } else {
            return forest.getSimpleDensity(point).distances;
        }
    }

    /**
     * same as getNewAttribution, except when just the score suffices
     * 
     * @param choice the mode in question
     * @param point  the point in question
     * @param forest the resident RCF
     * @return the score corresponding to the mode
     */
    double getNewScore(int choice, float[] point, RandomCutForest forest) {
        if (choice == EXPECTED_INVERSE_DEPTH_INDEX) {
            return forest.getAnomalyScore(point);
        } else {
            return forest.getSimpleDensity(point).distances.getHighLowSum();
        }
    }

    /**
     * returns the threshold and grade corresponding to a mode choice (based on
     * scoring strategy) currently the scoring strategy is unused, but would likely
     * be used in future
     * 
     * @param strategy    the scoring strategy
     * @param choice      the chosen mode
     * @param scoreVector the vector of scores
     * @param method      the transformation method used
     * @param dimension   the number of dimensions in RCF (used in auto adjustment
     *                    of thresholds)
     * @param shingleSize the shingle size (used in auto adjustment of thresholds)
     * @return a weighted object where the index is the threshold and the weight is
     *         the grade
     */
    protected Weighted<Double> getThresholdAndGrade(ScoringStrategy strategy, int choice, double[] scoreVector,
            TransformMethod method, int dimension, int shingleSize) {
        if (choice == EXPECTED_INVERSE_DEPTH_INDEX) {
            return thresholders[EXPECTED_INVERSE_DEPTH_INDEX]
                    .getThresholdAndGrade(scoreVector[EXPECTED_INVERSE_DEPTH_INDEX], method, dimension, shingleSize);
        } else {
            return thresholders[DISTANCE_INDEX].getPrimaryThresholdAndGrade(scoreVector[DISTANCE_INDEX]);
        }
    }

    /**
     * the strategy to save scores based on the scoring strategy
     * 
     * @param strategy       the strategy
     * @param choice         the mode for which corrected score applies
     * @param scoreVector    the vector of scores
     * @param correctedScore the estimated score with corrections (can be the same
     *                       as score)
     * @param method         the transformation method used
     * @param shingleSize    the shingle size
     */
    protected void saveScores(ScoringStrategy strategy, int choice, double[] scoreVector, double correctedScore,
            TransformMethod method, int shingleSize) {

        if (scoreVector[EXPECTED_INVERSE_DEPTH_INDEX] > 0) {
            double temp = (choice == EXPECTED_INVERSE_DEPTH_INDEX) ? correctedScore
                    : scoreVector[EXPECTED_INVERSE_DEPTH_INDEX];
            double last = (strategy == lastStrategy) ? lastScore[EXPECTED_INVERSE_DEPTH_INDEX] : 0;
            thresholders[EXPECTED_INVERSE_DEPTH_INDEX].update(scoreVector[EXPECTED_INVERSE_DEPTH_INDEX], temp, last,
                    method);
        }
        if (scoreVector[DISTANCE_INDEX] > 0) {
            thresholders[DISTANCE_INDEX].update(scoreVector[DISTANCE_INDEX], lastScore[DISTANCE_INDEX]);
        }
        if (shingleSize > 1) {
            for (int i = 0; i < NUMBER_OF_MODES; i++) {
                lastScore[i] = scoreVector[i];
            }
        }
    }

    /**
     * the core of the predictor-corrector thresholding for shingled data points. It
     * uses a simple threshold provided by the basic thresholder. It first checks if
     * obvious effects of the present; and absent such, for repeated breaches, how
     * critical is the new current information
     *
     * @param result                    returns the augmented description
     * @param lastSignificantDescriptor state of the computation for the last
     *                                  candidate anomaly
     * @param forest                    the resident RCF
     * @return the anomaly descriptor result (which has plausibly mutated)
     */
    protected <P extends AnomalyDescriptor> P detect(P result, RCFComputeDescriptor lastSignificantDescriptor,
            RandomCutForest forest) {
        if (result.getRCFPoint() == null) {
            return result;
        }
        float[] point = result.getRCFPoint();
        ScoringStrategy strategy = result.getScoringStrategy();
        double[] scoreVector = new double[NUMBER_OF_MODES];
        DiVector[] attributionVector = new DiVector[NUMBER_OF_MODES];

        final int originalChoice = populateScores(strategy, point, forest, scoreVector, attributionVector);

        DiVector attribution = null;
        final double score = scoreVector[originalChoice];

        // we will not alter the basic score from RCF under any circumstance
        result.setRCFScore(score);

        // we will not have zero scores affect any thresholding
        if (score == 0) {
            return result;
        }

        long internalTimeStamp = result.getInternalTimeStamp();

        int shingleSize = result.getShingleSize();

        Weighted<Double> thresholdAndGrade = getThresholdAndGrade(strategy, originalChoice, scoreVector,
                result.getTransformMethod(), point.length, shingleSize);
        final double originalThreshold = thresholdAndGrade.index;
        double workingThreshold = originalThreshold;
        double workingGrade = thresholdAndGrade.weight;
        // we will not alter this
        result.setThreshold(originalThreshold);

        boolean candidate = false;

        if (workingGrade > 0 && lastDescriptor != null) {
            if (score > lastDescriptor.getRCFScore()) {
                candidate = true;
            }
            double runDiscount = max(workingThreshold, lastDescriptor.getThreshold())
                    * (1 + max(0.2, runLength / (2.0 * max(10, shingleSize))));
            if (lastDescriptor.getRCFScore() - lastDescriptor.getThreshold() > score - runDiscount) {
                // the 'run' or the sequence of observations that create large scores
                // because of data (concept?) drift is defined to increase permissively
                // so that it is clear when the threshold is above the scores
                // a consequence of this can be masking -- anomalies just after a run/drift
                // would be difficult to determine -- but those should be difficult to determine
                candidate = true;
            }
        }

        if (workingGrade > 0 && strategy == ScoringStrategy.MULTI_MODE) {
            Weighted<Double> temp = thresholders[DISTANCE_INDEX]
                    .getPrimaryThresholdAndGrade(scoreVector[DISTANCE_INDEX]);
            if (temp.index > 0 && temp.weight == 0) {
                // there is a valid threshold and the grade is 0
                workingGrade = 0;
                result.setCorrectionMode(CorrectionMode.MULTI_MODE);
            }
        }

        if (lastDescriptor != null && lastDescriptor.getExpectedRCFPoint() != null) {
            lastSignificantDescriptor = lastDescriptor;
        }

        int gap = (int) (internalTimeStamp - lastSignificantDescriptor.getInternalTimeStamp());
        int difference = gap * baseDimension;

        float[] correctedPoint = null;
        double correctedScore = score;
        float[] expectedPoint = null;
        boolean inHighScoreRegion = false;
        int index = 0;
        int relative = (gap >= shingleSize) ? -shingleSize : -gap;

        int choice = originalChoice;
        if (strategy == ScoringStrategy.MULTI_MODE_RECALL && workingGrade == 0 && gap >= shingleSize) {
            // if overlapping shingles are being ruled out, then reconsidering those may not
            // be useful
            Weighted<Double> temp = thresholders[DISTANCE_INDEX]
                    .getPrimaryThresholdAndGrade(scoreVector[DISTANCE_INDEX]);
            choice = DISTANCE_INDEX;
            correctedScore = scoreVector[DISTANCE_INDEX];
            workingGrade = temp.weight;
            workingThreshold = temp.index;
        }

        // we perform basic correction
        correctedPoint = applyPastCorrector(point, gap, shingleSize, point.length / shingleSize, result.getScale(),
                result.getTransformMethod(), lastSignificantDescriptor);

        /**
         * we check if the point is too close to 0 for centered transforms as well as
         * explainable by the default distribution of differences this acts as a filter
         * and an upper bound for the grade
         */
        if (workingGrade > 0) {
            workingGrade *= centeredTransformPass(result, correctedPoint);
            if (workingGrade == 0) {
                result.setCorrectionMode(CorrectionMode.NOISE);
            }
        }

        if (workingGrade > 0) {
            inHighScoreRegion = true;

            if (!Arrays.equals(correctedPoint, point)) {
                attribution = getNewAttribution(choice, correctedPoint, forest);
                correctedScore = attribution.getHighLowSum();
                if (correctedScore > workingThreshold) {
                    int tempIndex = maxContribution(attribution, point.length / shingleSize, relative) + 1;
                    // use the additional new data for explanation
                    int tempStartPosition = point.length + (tempIndex - 1) * point.length / shingleSize;
                    float[] tempPoint = getExpectedPoint(attribution, tempStartPosition, point.length / shingleSize,
                            correctedPoint, forest);
                    if (tempPoint != null) {
                        DiVector tempAttribution = getNewAttribution(choice, tempPoint, forest);
                        correctedScore = tempAttribution.getHighLowSum();
                        if (!trigger(attribution, difference, point.length / shingleSize, tempAttribution,
                                lastSignificantDescriptor, workingThreshold)) {
                            workingGrade = 0;
                            result.setCorrectionMode(CorrectionMode.ANOMALY_IN_SHINGLE);
                        }
                    }
                }
            } else {
                attribution = getCachedAttribution(choice, point, attributionVector, forest);
            }

            if (workingGrade > 0 && result.getScale() != null) {
                index = (shingleSize == 1) ? 0 : maxContribution(attribution, point.length / shingleSize, relative) + 1;

                int startPosition = point.length + (index - 1) * point.length / shingleSize;
                DiVector uncertaintyBox = constructUncertaintyBox(point, startPosition, result);

                if (autoAdjust && explainedByConditionalField(uncertaintyBox, point, correctedPoint, startPosition,
                        result, forest)) {
                    workingGrade = 0;
                    result.setCorrectionMode(CONDITIONAL_FORECAST);
                } else {

                    expectedPoint = getExpectedPoint(attribution, startPosition, point.length / shingleSize,
                            correctedPoint, forest);
                    if (expectedPoint != null) {
                        if (difference < point.length) {
                            DiVector newAttribution = getNewAttribution(choice, expectedPoint, forest);
                            correctedScore = newAttribution.getHighLowSum();
                            if (!trigger(attribution, difference, point.length / shingleSize, newAttribution,
                                    lastSignificantDescriptor, workingThreshold)) {
                                workingGrade = 0;
                                result.setCorrectionMode(CorrectionMode.ANOMALY_IN_SHINGLE);
                            }
                        } else {
                            // attribution will not be used
                            correctedScore = getNewScore(choice, point, forest);
                        }

                        if (workingGrade > 0 && withinGap(uncertaintyBox, startPosition, result.getScale(), point,
                                expectedPoint, point.length / shingleSize)) {
                            workingGrade = 0;
                            result.setCorrectionMode(CorrectionMode.FORECAST);
                        }
                    }
                }
            }
            if (workingGrade == 0) {
                // note score is the original score
                correctedScore = score;
            }
        }

        if (candidate) {
            if (autoAdjust) {
                for (int y = 0; y < baseDimension; y++) {
                    deviationsActual[y].update(point[point.length - baseDimension + y]);
                    if (expectedPoint != null) {
                        deviationsExpected[y].update(expectedPoint[point.length - baseDimension + y]);
                    }
                }
                if (runLength > DEFAULT_RUN_ALLOWED && workingGrade > 0) {
                    boolean within = true;
                    for (int y = 0; y < baseDimension && within; y++) {
                        within = Math
                                .abs(deviationsActual[y].getMean() - point[point.length - baseDimension + y]) < max(
                                        2 * deviationsActual[y].getDeviation(),
                                        // results are in original space -- need to scale back to RCF space
                                        noiseFactor * result.getDeviations()[baseDimension + y] / result.getScale()[y]);
                        // estimation of noise from within the run as well as a long term estimation
                        if (expectedPoint != null) {
                            double u = Math.abs(
                                    deviationsExpected[y].getMean() - expectedPoint[point.length - baseDimension + y]);
                            within = within && Math.abs(deviationsExpected[y].getMean()
                                    - expectedPoint[point.length - baseDimension + y]) < 2
                                            * max(deviationsExpected[y].getDeviation(),
                                                    deviationsActual[y].getDeviation())
                                            + 0.1 * Math.abs(
                                                    deviationsActual[y].getMean() - deviationsExpected[y].getMean());
                            // forecasts cannot be more accurate than actuals; and forecasting would
                            // not be exact
                        }
                    }
                    if (within) {
                        result.setCorrectionMode(DATA_DRIFT);
                        workingGrade = 0;
                    }
                }
            }
            if (ignoreDrift && workingGrade > 0) {
                if (runLength > 0 && gap < shingleSize) {
                    result.setCorrectionMode(ALERT_ONCE);
                    workingGrade = 0;
                }
            }
        }

        result.setAnomalyGrade(workingGrade);
        result.setInHighScoreRegion(inHighScoreRegion);

        if (workingGrade > 0) {
            if (expectedPoint != null) {
                result.setExpectedRCFPoint(expectedPoint);
            }
            attribution.renormalize(result.getRCFScore());
            result.setStartOfAnomaly(true);
            result.setAttribution(attribution);
            result.setRelativeIndex(index);
            ++runLength;
        } else if (result.getCorrectionMode() == NONE) {
            runLength = 0;
            if (autoAdjust) {
                for (int y = 0; y < baseDimension; y++) {
                    deviationsActual[y].reset();
                    deviationsExpected[y].reset();
                }
            }
        } else if (runLength > 0) {
            // cannot start a run; but the run can be sustained
            ++runLength;
        }

        lastDescriptor = result.copyOf();
        saveScores(strategy, choice, scoreVector, correctedScore, result.getTransformMethod(), shingleSize);
        return result;
    }

    public void setZfactor(double factor) {
        for (int i = 0; i < thresholders.length; i++) {
            thresholders[i].setZfactor(factor);
        }
    }

    public void setAbsoluteThreshold(double lower) {
        // only applies to thresholder 0
        thresholders[EXPECTED_INVERSE_DEPTH_INDEX].setAbsoluteThreshold(lower);
    }

    public void setScoreDifferencing(double persistence) {
        // only applies to thresholder 0
        thresholders[EXPECTED_INVERSE_DEPTH_INDEX].setScoreDifferencing(persistence);
    }

    public void setInitialThreshold(double initial) {
        // only applies to thresholder 0
        thresholders[EXPECTED_INVERSE_DEPTH_INDEX].setInitialThreshold(initial);
    }

    public void setNumberOfAttributors(int numberOfAttributors) {
        checkArgument(numberOfAttributors > 0, "cannot be negative");
        this.numberOfAttributors = numberOfAttributors;
    }

    public int getNumberOfAttributors() {
        return numberOfAttributors;
    }

    public double[] getLastScore() {
        return lastScore;
    }

    public void setLastScore(double[] score) {
        if (score != null) {
            System.arraycopy(score, 0, lastScore, 0, min(NUMBER_OF_MODES, score.length));
        }
    }

    void validateIgnore(double[] shift, int length) {
        checkArgument(shift.length == length, () -> "has to be of length " + 4 * baseDimension);
        for (double element : shift) {
            checkArgument(element >= 0, "has to be non-negative");
        }
    }

    public void setIgnoreNearExpectedFromAbove(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift, baseDimension);
            System.arraycopy(ignoreSimilarShift, 0, ignoreNearExpectedFromAbove, 0, baseDimension);
        }
    }

    public void setIgnoreNearExpectedFromBelow(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift, baseDimension);
            System.arraycopy(ignoreSimilarShift, 0, ignoreNearExpectedFromBelow, 0, baseDimension);
        }
    }

    public void setIgnoreNearExpectedFromAboveByRatio(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift, baseDimension);
            System.arraycopy(ignoreSimilarShift, 0, ignoreNearExpectedFromAboveByRatio, 0, baseDimension);
        }
    }

    public void setIgnoreNearExpectedFromBelowByRatio(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift, baseDimension);
            System.arraycopy(ignoreSimilarShift, 0, ignoreNearExpectedFromBelowByRatio, 0, baseDimension);
        }
    }

    // to be used for the state classes only
    public void setIgnoreNearExpected(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift, 4 * baseDimension);
            System.arraycopy(ignoreSimilarShift, 0, ignoreNearExpectedFromAbove, 0, baseDimension);
            System.arraycopy(ignoreSimilarShift, baseDimension, ignoreNearExpectedFromBelow, 0, baseDimension);
            System.arraycopy(ignoreSimilarShift, 2 * baseDimension, ignoreNearExpectedFromAboveByRatio, 0,
                    baseDimension);
            System.arraycopy(ignoreSimilarShift, 3 * baseDimension, ignoreNearExpectedFromBelowByRatio, 0,
                    baseDimension);
        }
    }

    public double[] getIgnoreNearExpected() {
        double[] answer = new double[4 * baseDimension];
        System.arraycopy(ignoreNearExpectedFromAbove, 0, answer, 0, baseDimension);
        System.arraycopy(ignoreNearExpectedFromBelow, 0, answer, baseDimension, baseDimension);
        System.arraycopy(ignoreNearExpectedFromAboveByRatio, 0, answer, 2 * baseDimension, baseDimension);
        System.arraycopy(ignoreNearExpectedFromBelowByRatio, 0, answer, 3 * baseDimension, baseDimension);
        return answer;
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public BasicThresholder[] getThresholders() {
        return thresholders;
    }

    public int getBaseDimension() {
        return baseDimension;
    }

    public ScoringStrategy getLastStrategy() {
        return lastStrategy;
    }

    public void setLastStrategy(ScoringStrategy strategy) {
        this.lastStrategy = strategy;
    }

    public Deviation[] getDeviations() {
        if (!autoAdjust) {
            return null;
        }
        checkArgument(deviationsActual.length == deviationsExpected.length, "incorrect state");
        checkArgument(deviationsActual.length == baseDimension, "length should be base dimension");

        Deviation[] answer = new Deviation[2 * deviationsActual.length];
        for (int i = 0; i < deviationsActual.length; i++) {
            answer[i] = deviationsActual[i];
        }
        for (int i = 0; i < deviationsExpected.length; i++) {
            answer[i + deviationsActual.length] = deviationsExpected[i];
        }
        return answer;
    }

    public double getSamplingRate() {
        return samplingRate;
    }

    public void setSamplingRate(double samplingRate) {
        checkArgument(samplingRate > 0, " cannot be negative");
        checkArgument(samplingRate < 1.0, " has to be in [0,1)");
        this.samplingRate = samplingRate;
    }

    public double[] getModeInformation() {
        return modeInformation;
    }

    // to be used in future
    public void setModeInformation(double[] modeInformation) {
    }

    public boolean isAutoAdjust() {
        return autoAdjust;
    }

    public void setAutoAdjust(boolean autoAdjust) {
        this.autoAdjust = autoAdjust;
    }

    public double getNoiseFactor() {
        return noiseFactor;
    }

    public void setNoiseFactor(double noiseFactor) {
        this.noiseFactor = noiseFactor;
    }

    public void setIgnoreDrift(boolean ignoreDrift) {
        this.ignoreDrift = ignoreDrift;
    }

    public boolean isIgnoreDrift() {
        return ignoreDrift;
    }

    public void setLastDescriptor(RCFComputeDescriptor lastDescriptor) {
        this.lastDescriptor = lastDescriptor.copyOf();
    }

    public RCFComputeDescriptor getLastDescriptor() {
        return lastDescriptor;
    }

    public int getRunLength() {
        return runLength;
    }

    public void setRunLength(int runLength) {
        this.runLength = runLength;
    }

    public double getSamplingSupport() {
        return samplingSupport;
    }

    public void setSamplingSupport(double sampling) {
        checkArgument(sampling >= 0, " cannot be negative ");
        checkArgument(sampling < 2 * DEFAULT_SAMPLING_SUPPORT,
                " cannot be more than " + (2 * DEFAULT_SAMPLING_SUPPORT));
    }
}
