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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.util.Weighted;

import java.util.Arrays;
import java.util.Random;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor.DEFAULT_NORMALIZATION_PRECISION;
import static java.lang.Math.min;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
public class PredictorCorrector {
    private static double DEFAULT_DIFFERENTIAL_FACTOR = 0.3;

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 5;

    public static double DEFAULT_MULTI_MODE_SAMPLING_RATE = 0.1;

    protected static int NUMBER_OF_MODES = 2;

    // the following vectors enable suppression of anomalies
    // the first pair correspond to additive differences
    // the second pair correspond to multiplicative differences
    // multiplicative differences are not meaningful for differenced operations

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

    protected Deviation[] deviationsAbove;

    protected Deviation[] deviationsBelow;

    protected double samplingRate = DEFAULT_MULTI_MODE_SAMPLING_RATE;

    protected boolean autoAdjust = false;

    public PredictorCorrector(double timeDecay, double anomalyRate, boolean adjustThresholds, boolean adjust, int baseDimension,
            long randomSeed) {
        this.thresholders = new BasicThresholder[NUMBER_OF_MODES];
        thresholders[0] = new BasicThresholder(timeDecay, anomalyRate, adjustThresholds);
        thresholders[1] = new BasicThresholder(timeDecay);
        this.baseDimension = baseDimension;
        this.randomSeed = randomSeed;
        this.deviationsAbove = new Deviation[baseDimension];
        this.deviationsBelow = new Deviation[baseDimension];
        this.autoAdjust = adjust;
        if (adjust) {
            for(int i = 0;i < baseDimension;i++){
                this.deviationsAbove[i] = new Deviation(timeDecay);
                this.deviationsBelow[i] = new Deviation(timeDecay);
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
        checkArgument(deviations == null || deviations.length == 2 * baseDimension, "incorrect state");
        this.thresholders = new BasicThresholder[NUMBER_OF_MODES];
        int size = min(thresholders.length, NUMBER_OF_MODES);
        for (int i = 0; i < size; i++) {
            this.thresholders[i] = thresholders[i];
        }
        for (int i = size; i < NUMBER_OF_MODES; i++) {
            this.thresholders[i] = new BasicThresholder(thresholders[0].getPrimaryDeviation().getDiscount());
        }
        this.deviationsAbove = new Deviation[baseDimension];
        this.deviationsBelow = new Deviation[baseDimension];
        if (deviations != null) {
            for (int i = 0; i < baseDimension; i++) {
                deviationsAbove[i] = deviations[i];
            }
            for (int i = 0; i < baseDimension; i++) {
                deviationsBelow[i] = deviations[i + baseDimension];
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

    /** the following creates the expected poin based on RCF forecasting
     * @param diVector the attribution vector that is used to choose which elements are to be predicted
     * @param position the block of (multivariate) elements we are focusing on
     * @param baseDimension the base dimension of the block
     * @param point the point near which we wish to predict
     * @param forest the resident RCF
     * @return a vector that is most likely, conditioned on changing a few elements in
     * the block at position
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

            if (pick > DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS) {
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
     * in a high score region with a previous anomalies, we use this to determine if
     * the "residual contribution" since the last anomaly would have sufficed to
     * trigger anomaly designation on its own.
     */
    /**
     * a subroutine that helps eliminates flagging anomalies too close to a previously flagged
     * anomaly -- this avoids the repetition due to shingling; but still can detect some anomalies
     * if the deviations are ususual
     * @param candidate the candidate attribution of the point
     * @param difference the shift in term of basic elements from the last anomaly
     * @param baseDimension the size of a block
     * @param ideal an idealized version of the candidate (can be null) where the most offending
     *              elements are imputed out
     * @param lastAnomalyDescriptor the description of the last anomaly
     * @param workingThreshold the threshold to exceed
     * @return true if the candidate is sufficiently different and false otherwise
     */

    protected boolean trigger(DiVector candidate, int difference, int baseDimension, DiVector ideal,
            IRCFComputeDescriptor lastAnomalyDescriptor, double workingThreshold) {
        int dimensions = candidate.getDimensions();
        DiVector lastAnomalyAttribution = lastAnomalyDescriptor.getAttribution();
        if (lastAnomalyAttribution == null || difference >= dimensions) {
            return true;
        }
        checkArgument(lastAnomalyAttribution.getDimensions() == dimensions, " error in DiVectors");
        int shingleSize = dimensions / baseDimension;

        if (ideal == null) {
            double remainder = 0;
            for (int i = dimensions - difference; i < dimensions; i++) {
                remainder += candidate.getHighLowSum(i);
            }
            // simplifying the following since remainder * dimensions/difference corresponds
            // to the
            // impact of the new data since the last anomaly
            return remainder * dimensions / difference > workingThreshold;
        } else {
            double lastAnomalyScore = lastAnomalyDescriptor.getRCFScore();
            double differentialRemainder = 0;
            for (int i = dimensions - difference; i < dimensions; i++) {
                differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i])
                        + Math.abs(candidate.high[i] - ideal.high[i]);
            }
            return (differentialRemainder > DEFAULT_DIFFERENTIAL_FACTOR * lastAnomalyScore)
                    && differentialRemainder * dimensions / difference > workingThreshold;
        }
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
     * @param lastAnomalyDescriptor description of the last anomaly
     * @return the corrected point
     */
    float[] applyBasicCorrector(float[] point, int gap, int shingleSize, int baseDimensions,
            IRCFComputeDescriptor lastAnomalyDescriptor) {
        checkArgument(gap >= 0 && gap <= shingleSize, "incorrect invocation");
        float[] correctedPoint = Arrays.copyOf(point, point.length);
        float[] lastExpectedPoint = toFloatArray(lastAnomalyDescriptor.getExpectedRCFPoint());
        double[] lastAnomalyPoint = lastAnomalyDescriptor.getRCFPoint();
        int lastRelativeIndex = lastAnomalyDescriptor.getRelativeIndex();
        if (gap < shingleSize) {
            System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                    point.length - gap * baseDimensions);
        }
        if (lastRelativeIndex == 0) { // it is possible to fix other cases, but is more complicated
            TransformMethod transformMethod = lastAnomalyDescriptor.getTransformMethod();
            if (transformMethod == TransformMethod.DIFFERENCE
                    || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
                for (int y = 0; y < baseDimensions; y++) {
                    correctedPoint[point.length - gap * baseDimensions
                            + y] += lastAnomalyPoint[point.length - baseDimensions + y]
                                    - lastExpectedPoint[point.length - baseDimensions + y];
                }
            } else if (lastAnomalyDescriptor.getForestMode() == ForestMode.TIME_AUGMENTED) {
                // definitely correct the time dimension which is always differenced
                // this applies to the non-differenced cases
                correctedPoint[point.length - (gap - 1) * baseDimensions - 1] += lastAnomalyPoint[point.length - 1]
                        - lastExpectedPoint[point.length - 1];

            }
        }
        return correctedPoint;
    }

    /** The following is useful for managing late detection of anomalies -- this calculates the
     * zig-zag over the values in the late detection
     * @param point the point being scored
     * @param startPosition the position of the block where we think the anomaly started
     * @param index the specific index in the block being tracked
     * @param baseDimension the size of the block
     * @param differenced has differencing been performed already
     * @return the L1 deviation
     */
    double calculatePathDeviation(float[] point, int startPosition, int index, int baseDimension, boolean differenced) {
        int position = startPosition;
        double variation = 0;
        while (position + index + baseDimension < point.length) {
            variation += (differenced) ? Math.abs(point[position + index])
                    : Math.abs(point[position + index] - point[position + baseDimension + index]);
            position += baseDimension;
        }
        return variation;
    }

    /**
     * the following function determines if the difference is significant. the
     * difference could arise from the floating point operations in the
     * transformation. Independently small variations determined by the dynamically
     * configurable setting of ignoreSimilarFactor may not be interest.
     *
     * @param significantScore If the score is significant (but it is possible that
     *                         we ignore anomalies below/above expected and hence
     *                         this bit can be modified)
     * @param point            the current point in question (in RCF space)
     * @param newPoint         the expected point (in RCF space)
     * @param startPosition    the (most likely) location of the anomaly
     * @param result           the result to be vended
     * @return true if the changes are significant (hence an anomaly) and false
     *         otherwise
     */

    protected <P extends AnomalyDescriptor> boolean isSignificant(boolean significantScore, float[] point,
            float[] newPoint, int startPosition, P result) {
        checkArgument(point.length == newPoint.length, "incorrect invocation");
        int baseDimensions = result.getDimension() / result.getShingleSize();
        TransformMethod method = result.getTransformMethod();
        boolean differenced = (method == TransformMethod.DIFFERENCE)
                || (method == TransformMethod.NORMALIZE_DIFFERENCE);
        double[] scale = result.getScale();
        double[] shift = result.getShift();
        // backward compatibility of older models
        if (scale == null || shift == null) {
            return true;
        }
        boolean answer = false;
        // only for input dimensions, for which scale is defined currently
        for (int y = 0; y < baseDimension && !answer; y++) {
            double observedGap = Math.abs(point[startPosition + y] - newPoint[startPosition + y]);
            double pathGap = calculatePathDeviation(point, startPosition, y, baseDimensions, differenced);
            if (observedGap > 0.1 * pathGap) {
                double scaleFactor = (scale == null) ? 1.0 : scale[y];
                double delta = observedGap * scaleFactor;
                double shiftBase = (shift == null) ? 0 : shift[y];
                double shiftAmount = 0;

                // the conditional below is redundant, since abs(shiftBase) is being multiplied
                // but kept as a placeholder for tuning constants if desired
                if (shiftBase != 0) {
                    double multiplier = (method == TransformMethod.NORMALIZE) ? 4 : 2;
                    shiftAmount += multiplier * DEFAULT_NORMALIZATION_PRECISION * Math.abs(shiftBase);
                }

                // note that values cannot be reconstructed well if differencing was invoked
                double a = Math.abs(scaleFactor * point[startPosition + y] + shiftBase);
                double b = Math.abs(scaleFactor * newPoint[startPosition + y] + shiftBase);

                // for non-trivial transformations -- both transformations are used enable
                // relative error
                // the only transformation currently ruled out is TransformMethod.NONE
                if (scaleFactor != 1.0 || shiftBase != 0) {
                    double multiplier = (method == TransformMethod.NORMALIZE) ? 2 : 1;
                    shiftAmount += multiplier * DEFAULT_NORMALIZATION_PRECISION * (scaleFactor + (a + b) / 2);
                }
                answer = significantScore || (delta > shiftAmount + DEFAULT_NORMALIZATION_PRECISION);
                if (answer) {
                    answer = (a < b - ignoreNearExpectedFromBelow[y]) || (a > b + ignoreNearExpectedFromAbove[y]);
                }
            }
        }
        return answer;
    }

    /**
     * populates the scores and sets the score and attribution vectors; note some of
     * the attributions can remain null (for efficiency reasons)
     *
     * @param strategy          the scoring strategy
     * @param scoreVector       the vector of scores
     * @param attributionVector the vector of attributions
     * @return the index of the score/attribution that is relevant
     */

    protected int populateScores(ScoringStrategy strategy, float[] point, RandomCutForest forest, double[] scoreVector,
            DiVector[] attributionVector) {
        if (strategy != ScoringStrategy.DISTANCE) {
            scoreVector[0] = forest.getAnomalyScore(point);
            if (strategy == ScoringStrategy.MULTI_MODE || strategy == ScoringStrategy.MULTI_MODE_RECALL) {
                attributionVector[1] = forest.getSimpleDensity(point).distances;
                scoreVector[1] = attributionVector[1].getHighLowSum();
            }
            return 0;
        } else {
            attributionVector[1] = forest.getSimpleDensity(point).distances;
            scoreVector[1] = attributionVector[1].getHighLowSum();
            return 1;
        }
    }

    /**
     * returned the attribution vector; it tries to reused cached version to save computation
     * @param choice the mode of the attribution in question
     * @param point the point being considered
     * @param attributionVector the vector (cachee) of attributions
     * @param forest the resident RCF
     * @return the attribution correspond to the mode of attribution
     */
    DiVector getCachedAttribution(int choice, float[] point, DiVector[] attributionVector, RandomCutForest forest) {
        if (attributionVector[choice] == null) {
            if (choice == 0) {
                attributionVector[0] = forest.getAnomalyAttribution(point);
            } else {
                attributionVector[1] = forest.getSimpleDensity(point).distances;
            }
        }
        return attributionVector[choice];
    }

    /**
     * computes the attribution of a (candidate) point based on mode, when the results are not
     * expected to be cached
     * @param choice the mode
     * @param point the point in question
     * @param forest the resident RCF
     * @return the attribution of that mode
     */
    DiVector getNewAttribution(int choice, float[] point, RandomCutForest forest) {
        if (choice == 0) {
            return forest.getAnomalyAttribution(point);
        } else {
            return forest.getSimpleDensity(point).distances;
        }
    }

    /**
     * same as getNewAttribution, except when just the score suffices
     * @param choice the mode in question
     * @param point the point in question
     * @param forest the resident RCF
     * @return the score corresponding to the mode
     */
    double getNewScore(int choice, float[] point, RandomCutForest forest) {
        if (choice == 0) {
            return forest.getAnomalyScore(point);
        } else {
            return forest.getSimpleDensity(point).distances.getHighLowSum();
        }
    }

    /**
     * returns the threshold and grade corresponding to a mode choice (based on scoring strategy)
     * currently the scoring strategy is unusued, but would likely be used in future
     * @param strategy the scoring strategy
     * @param choice the chosen mode
     * @param scoreVector the vector of scores
     * @param method the transformation method used
     * @param dimension the number of dimensions in RCF (used in auto adjustment of thresholds)
     * @param shingleSize the shingle size (used in auto adjustment of thresholds)
     * @return a weighted object where the index is the threshold and the weight is the grade
     */
    protected Weighted<Double> getThresholdAndGrade(ScoringStrategy strategy, int choice, double[] scoreVector,
            TransformMethod method, int dimension, int shingleSize) {
        if (choice == 0) {
            return thresholders[0].getThresholdAndGrade(scoreVector[0], method, dimension, shingleSize);
        } else {
            return thresholders[1].getPrimaryThresholdAndGrade(scoreVector[1]);
        }
    }

    /**
     * In MULTI_MODE_RECALL, the specific mode can be reassigned and thus we would need to compute
     * score and thresholds
     * @param strategy the scoring strategy
     * @param choice the choice of the mode
     * @param point the point being rescored
     * @param forest the resident RCF
     * @param method the transformation method being used
     * @param shingleSize the shingle size
     * @return a weighted object where index corresponds to threshold and weight corresponds to grade
     */

    protected Weighted<Double> rescoredThresholdAndGrade(ScoringStrategy strategy, int choice, float[] point,
            RandomCutForest forest, TransformMethod method, int shingleSize) {
        if (choice == 0) {
            double score = forest.getAnomalyScore(point);
            return thresholders[0].getThresholdAndGrade(score, method, point.length, shingleSize);
        } else {
            double score = forest.getSimpleDensity(point).distances.getHighLowSum();
            return thresholders[1].getPrimaryThresholdAndGrade(score);
        }
    }

    /**
     * the strategy to save scores based on the scoring strategy
     * @param strategy the strategy
     * @param choice the mode for which corrected score applies
     * @param scoreVector the vector of scores
     * @param correctedScore the estimated score with corrections (can be the same as score)
     * @param method the transformation method used
     * @param shingleSize the shingle size
     */
    protected void saveScores(ScoringStrategy strategy, int choice, double[] scoreVector, double correctedScore,
            TransformMethod method, int shingleSize) {
        if (strategy == ScoringStrategy.MULTI_MODE || strategy == ScoringStrategy.MULTI_MODE_RECALL) {
            for (int i = 0; i < NUMBER_OF_MODES; i++) {
                double temp = (i == choice) ? correctedScore : scoreVector[i];
                thresholders[i].update(scoreVector[i], temp, lastScore[i], method);
                if (shingleSize > 1) {
                    lastScore[i] = scoreVector[i];
                }
            }
        } else {
            thresholders[choice].update(scoreVector[choice], correctedScore, lastScore[choice],
                    method);
            if (shingleSize > 1) {
                lastScore[choice] = scoreVector[choice];
            }
        }
    }

    /**
     * the core of the predictor-corrector thresholding for shingled data points. It
     * uses a simple threshold provided by the basic thresholder. It first checks if
     * obvious effects of the present; and absent such, for repeated breaches, how
     * critical is the new current information
     *
     * @param result                returns the augmented description
     * @param lastAnomalyDescriptor state of the computation for the last anomaly
     * @param forest                the resident RCF
     * @return the anomaly descriptor result (which has plausibly mutated)
     */
    protected <P extends AnomalyDescriptor> P detect(P result, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {
        if (result.getRCFPoint() == null) {
            return result;
        }
        float[] point = toFloatArray(result.getRCFPoint());
        ScoringStrategy strategy = result.getScoringStrategy();
        double[] scoreVector = new double[NUMBER_OF_MODES];
        DiVector[] attributionVector = new DiVector[NUMBER_OF_MODES];

        final int originalChoice = populateScores(strategy, point, forest, scoreVector, attributionVector);

        DiVector attribution = null;
        double score = scoreVector[originalChoice];

        // we will not alter the basic score from RCF under any circumstance
        result.setRCFScore(score);

        // we will not have zero scores affect any thresholding
        if (score == 0) {
            return result;
        }

        long internalTimeStamp = result.getInternalTimeStamp();

        int shingleSize = result.getShingleSize();
        int startPosition = (shingleSize - 1) * baseDimension;

        // we will adjust *both* the grade and the threshold

        double workingGrade = 0;
        double workingThreshold = 0;

        int gap = (int) (internalTimeStamp - lastAnomalyDescriptor.getInternalTimeStamp());
        int difference = gap * shingleSize;

        float[] correctedPoint = null;
        double correctedScore = score;
        boolean inHighScoreRegion = false;
        int index = 0;
        int relative = (gap >= shingleSize) ? -shingleSize : -gap;

        Weighted<Double> thresholdAndGrade = getThresholdAndGrade(strategy, originalChoice, scoreVector,
                result.transformMethod, point.length, shingleSize);
        workingThreshold = thresholdAndGrade.index;
        workingGrade = thresholdAndGrade.weight;

        if (strategy == ScoringStrategy.MULTI_MODE) {
            Weighted<Double> temp = thresholders[1].getPrimaryThresholdAndGrade(scoreVector[1]);
            if (temp.weight == 0) {
                workingGrade = 0;
            }
        }

        int choice = originalChoice;
        if (strategy == ScoringStrategy.MULTI_MODE_RECALL && workingGrade == 0) {
            Weighted<Double> temp = thresholders[1].getPrimaryThresholdAndGrade(scoreVector[1]);
            choice = 1;
            correctedScore = scoreVector[1];
            workingGrade = temp.weight;
        }

        /*
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         *
         * We then check if (1) we have another anomaly in the current shingle (2) have
         * predictions about what the values should have been and (3) replacing by those
         * "should have been" makes the anomaly score of the new shingled point low
         * enough to not be an anomaly. In this case we can "explain" the high score is
         * due to the past and do not need to vend anomaly -- because the most recent
         * point, on their own would not produce an anomalous shingle.
         *
         * However, the strategy is only executable if there are (A) sufficiently many
         * observations and (B) enough data in each time point such that the forecast is
         * reasonable. While forecasts can be corrected for very low shingle sizes and
         * say 1d input, the allure of RCF is in the multivariate case. Even for 1d, a
         * shingleSize of 4 or larger would produce reasonable forecast for the purposes
         * of anomaly detection.
         */

        if (workingGrade > 0) {
            inHighScoreRegion = true;
            // the forecast may not be reasonable with less data
            if (!result.isReasonableForecast()) {
                attribution = getCachedAttribution(choice, point, attributionVector, forest);

                if (!trigger(attribution, difference, point.length / shingleSize, null, lastAnomalyDescriptor,
                        workingThreshold)) {
                    workingGrade = 0;
                }
                ;
                index = (shingleSize == 1 && workingGrade > 0) ? 0
                        : maxContribution(attribution, point.length / shingleSize, relative) + 1;
            } else {

                // note that the following is bypassed for shingleSize = 1 because it would not
                // make sense to connect the current evaluation with a previous value
                if (lastAnomalyDescriptor.getRCFPoint() != null && shingleSize > 1
                        && lastAnomalyDescriptor.getExpectedRCFPoint() != null && gap > 0 && gap <= shingleSize) {
                    correctedPoint = applyBasicCorrector(point, gap, shingleSize, point.length / shingleSize,
                            lastAnomalyDescriptor);

                    Weighted<Double> rescoredThresholdAndGrade = rescoredThresholdAndGrade(strategy, choice,
                            correctedPoint, forest, result.transformMethod, shingleSize);
                    correctedScore = rescoredThresholdAndGrade.index;
                    double tempGrade = rescoredThresholdAndGrade.weight;
                    if (tempGrade <= 0) {
                        workingGrade = 0;
                    }
                }

                if (workingGrade > 0) {
                    attribution = getCachedAttribution(choice, point, attributionVector, forest);

                    DiVector newAttribution = null;

                    /*
                     * we now find the time slice, relative to the current time, which is indicative
                     * of the high score. relativeIndex = 0 is current time. It is negative if the
                     * most egregious attribution was due to the past values in the shingle
                     */

                    index = (shingleSize == 1) ? 0
                            : maxContribution(attribution, point.length / shingleSize, relative) + 1;

                    startPosition = point.length + (index - 1) * point.length / shingleSize;
                    correctedPoint = getExpectedPoint(attribution, startPosition, point.length / shingleSize, point,
                            forest);
                    if (correctedPoint != null) {
                        if (difference < point.length) {
                            newAttribution = getNewAttribution(choice, correctedPoint, forest);
                            correctedScore = newAttribution.getHighLowSum();
                        } else {
                            // attribution will not be used
                            correctedScore = getNewScore(choice, point, forest);
                        }
                    }

                    if (!trigger(attribution, difference, point.length / shingleSize, newAttribution,
                            lastAnomalyDescriptor, workingThreshold)) {
                        workingGrade = 0;
                    }
                    ;

                    if (workingGrade > 0 && correctedPoint != null) {
                        boolean significantScore = strategy == ScoringStrategy.DISTANCE || score > 1.5
                                || score > workingThreshold + 0.25
                                || (score > correctedScore + 0.25 && gap > shingleSize);
                        // significantScore is the signal sent; but can can be overruled by
                        // ignoreSimilarShift
                        if (!isSignificant(significantScore, point, correctedPoint, startPosition, result)) {
                            workingGrade = 0;
                        }
                        ;
                    }
                }
                if (workingGrade == 0) {
                    workingThreshold = score;
                    correctedScore = score;
                }
            }
        }

        result.setAnomalyGrade(workingGrade);
        result.setThreshold(workingThreshold);
        result.setInHighScoreRegion(inHighScoreRegion);

        if (workingGrade > 0) {
            if (correctedPoint != null) {
                result.setExpectedRCFPoint(toDoubleArray(correctedPoint));
            }
            if (choice != originalChoice) {
                attribution.renormalize(result.getRCFScore());
            }
            result.setAttribution(attribution);

            result.setRelativeIndex(index);
        }

        saveScores(strategy, choice, scoreVector, correctedScore, result.transformMethod, shingleSize);
        return result;
    }

    public void setZfactor(double factor) {
        for (int i = 0; i < thresholders.length; i++) {
            thresholders[i].setZfactor(factor);
        }
    }

    public void setLowerThreshold(double lower) {
        for (int i = 0; i < thresholders.length; i++) {
            thresholders[i].setLowerThreshold(lower);
        }
    }

    public void setScoreDifferencing(double persistence) {
        for (int i = 0; i < thresholders.length; i++) {
            thresholders[i].setScoreDifferencing(persistence);
        }
    }

    public void setInitialThreshold(double initial) {
        for (int i = 0; i < thresholders.length; i++) {
            thresholders[i].setInitialThreshold(initial);
        }
    }

    public void setNumberOfAttributors(int numberOfAttributors) {
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

    void validateIgnore(double[] shift) {
        checkArgument(shift.length == 4 * baseDimension, () -> "has to be of length " + 4 * baseDimension);
        for (double element : shift) {
            checkArgument(element >= 0, "has to be non-negative");
        }
    }

    public void setIgnoreNearExpected(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            validateIgnore(ignoreSimilarShift);
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
        if (!autoAdjust){
            return null;
        }
        checkArgument(deviationsAbove.length == deviationsBelow.length, "incorrect state");
        Deviation[] answer = new Deviation[2 * deviationsAbove.length];
        for (int i = 0; i < deviationsAbove.length; i++) {
            answer[i] = deviationsAbove[i];
        }
        for (int i = 0; i < deviationsBelow.length; i++) {
            answer[i + deviationsAbove.length] = deviationsBelow[i];
        }
        return answer;
    }

    public double getSamplingRate() {
        return samplingRate;
    }

    public void setSamplingRate(double samplingRate) {
        checkArgument(samplingRate > 0 && samplingRate < 1.0, " hast to be in [0,1)");
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

    public void setAutoAdjust(boolean autoAdjust){
        this.autoAdjust = autoAdjust;
    }
}
