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
import static com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor.DEFAULT_NORMALIZATION_PRECISION;

import java.util.Arrays;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.util.Weighted;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
public class PredictorCorrector {

    private static double DEFAULT_DIFFERENTIAL_FACTOR = 0.3;

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 5;

    double[] ignoreNearExpectedFromBelow;

    double[] ignoreNearExpectedFromAbove;

    // for anomaly description we would only look at these many top attributors
    // AExpected value is not well-defined when this number is greater than 1
    // that being said there is no formal restriction other than the fact that the
    // answers would be error prone as this parameter is raised.
    protected int numberOfAttributors = DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS;

    protected double lastScore = 0;

    protected BasicThresholder thresholder;

    // for mappers
    public PredictorCorrector(BasicThresholder thresholder, int baseDimension) {
        this.thresholder = thresholder;
        ignoreNearExpectedFromAbove = new double[baseDimension];
        ignoreNearExpectedFromBelow = new double[baseDimension];
    }

    public BasicThresholder getThresholder() {
        return thresholder;
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
     * produces the expected point (in the shingle space of RCF)
     * 
     * @param diVector      the attribution vector
     * @param position      the position to focus on (corresponding to the time
     *                      slice)
     * @param baseDimension the input dimension
     * @param point         the current point (in RCF shingled space)
     * @return the expected point -- it can be null if there are too many attibutors
     *         (thus confusing)
     */
    protected double[] getExpectedPoint(DiVector diVector, int position, int baseDimension, double[] point,
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
            while (pick < baseDimension && values[baseDimension - pick - 1] >= 0.1 * sum) {
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
     * 
     * @param candidate     attribution of the current point in consideration
     * @param gap           how long ago did the previous anomaly occur
     * @param baseDimension number of input attributes/variables (before shingling)
     * @param ideal         a form of expected attribution; can be null if there was
     *                      no previous anomaly in the shingle
     * @return true/false if the residual (extrapolated) score would trigger anomaly
     *         designation
     */
    protected boolean trigger(DiVector candidate, int gap, int baseDimension, DiVector ideal, TransformMethod method,
            IRCFComputeDescriptor lastAnomalyDescriptor) {
        DiVector lastAnomalyAttribution = lastAnomalyDescriptor.getAttribution();
        double lastAnomalyScore = lastAnomalyDescriptor.getRCFScore();
        if (lastAnomalyAttribution == null) {
            return true;
        }
        checkArgument(lastAnomalyAttribution.getDimensions() == candidate.getDimensions(), " error in DiVectors");
        int dimensions = candidate.getDimensions();

        int difference = baseDimension * gap;

        if (difference < dimensions) {
            if (ideal == null) {
                double remainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    remainder += candidate.getHighLowSum(i);
                }
                // simplifying the following since remainder * dimensions/difference corresponds
                // to the
                // impact of the new data since the last anomaly
                return thresholder.getThresholdAndGrade(remainder * dimensions / difference, method,
                        baseDimension).weight > 0;
            } else {
                double differentialRemainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i])
                            + Math.abs(candidate.high[i] - ideal.high[i]);
                }
                return (differentialRemainder > DEFAULT_DIFFERENTIAL_FACTOR * lastAnomalyScore)
                        && thresholder.getThresholdAndGrade(differentialRemainder * dimensions / difference, method,
                                baseDimension).weight > 0;
            }
        } else {
            return true;
        }

    }

    /**
     * a first stage corrector that attempts to fix the after effects of a previous
     * anomaly which may be in the shingle, or just preceding the shingle
     * 
     * @param point          the current (transformed) point under evaluation
     * @param gap            the relative position of the previous anomaly being
     *                       corrected
     * @param shingleSize    size of the shingle
     * @param baseDimensions number of dimensions in each shingle
     * @return the score of the corrected point
     */
    double[] applyBasicCorrector(double[] point, int gap, int shingleSize, int baseDimensions,
            IRCFComputeDescriptor lastAnomalyDescriptor) {
        checkArgument(gap >= 0 && gap <= shingleSize, "incorrect invocation");
        double[] correctedPoint = Arrays.copyOf(point, point.length);
        double[] lastExpectedPoint = lastAnomalyDescriptor.getExpectedRCFPoint();
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
     * @param baseDimensions   the number of dimensions before shingling
     * @param result           the result to be vended
     * @return true if the changes are significant (hence an anomaly) and false
     *         otherwise
     */

    protected <P extends AnomalyDescriptor> boolean isSignificant(boolean significantScore, double[] point,
            double[] newPoint, int startPosition, int baseDimensions, P result) {
        checkArgument(point.length == newPoint.length, "incorrect invocation");
        double[] scale = result.getScale();
        double[] shift = result.getShift();
        TransformMethod method = result.getTransformMethod();
        if (scale == null || shift == null) {
            return true;
        }
        boolean answer = false;
        for (int y = 0; y < baseDimensions && !answer; y++) {
            double scaleFactor = (scale == null) ? 1.0 : scale[y];
            double delta = Math.abs(point[startPosition + y] - newPoint[startPosition + y]) * scaleFactor;
            double shiftBase = (shift == null) ? 0 : shift[y];
            double shiftAmount = 0;
            if (scaleFactor != 1.0 || shiftBase != 0) {
                double multiplier = (method == TransformMethod.NORMALIZE) ? 4 : 2;
                shiftAmount += multiplier * DEFAULT_NORMALIZATION_PRECISION * Math.abs(shiftBase);
            }

            // note that values cannot be reconstructed well if differencing was invoked
            double a = Math.abs(scaleFactor * point[startPosition + y] + shiftBase);
            double b = Math.abs(scaleFactor * newPoint[startPosition + y] + shiftBase);

            // for non-trivial transformations
            if (scaleFactor != 1.0 || shiftBase != 0) {
                double multiplier = (method == TransformMethod.NORMALIZE) ? 2 : 1;
                shiftAmount += multiplier * DEFAULT_NORMALIZATION_PRECISION * (scaleFactor + (a + b) / 2);
            }
            answer = significantScore || (delta > shiftAmount + DEFAULT_NORMALIZATION_PRECISION);
            if (answer) {
                answer = (a < b - ignoreNearExpectedFromBelow[y]) || (a > b + ignoreNearExpectedFromAbove[y]);
            }
        }
        return answer;
    }

    /**
     * the core of the predictor-corrector thresholding for shingled data points. It
     * uses a simple threshold provided by the basic thresholder. It first checks if
     * obvious effects of the present; and absent such, for repeated breaches, how
     * critical is the new current information
     * 
     * @param result                returns the augmented description
     * @param lastAnomalyDescriptor state of the computation for the last anomaly
     * @return the anomaly descriptor result (which has plausibly mutated)
     */
    protected <P extends AnomalyDescriptor> P detect(P result, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {
        double[] point = result.getRCFPoint();
        if (point == null) {
            return result;
        }
        double score = 0;
        DiVector attribution = null;
        if (result.forestMode != ForestMode.DISTANCE) {
            score = forest.getAnomalyScore(point);
        } else {
            attribution = forest.getSimpleDensity(point).distances;
            score = attribution.getHighLowSum();
        }

        // we will not alter the basic score from RCF under any circumstance
        result.setRCFScore(score);
        result.setRCFPoint(point);

        long internalTimeStamp = result.getInternalTimeStamp();

        if (score == 0) {
            return result;
        }

        int shingleSize = (result.getDimension() == result.getInputLength()) ? 1 : result.getShingleSize();
        int baseDimensions = result.getDimension() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;

        // we will adjust *both* the grade and the threshold

        double workingGrade = 0;
        double workingThreshold = 0;
        if (result.forestMode != ForestMode.DISTANCE) {
            Weighted<Double> thresholdAndGrade = thresholder.getThresholdAndGrade(score, result.transformMethod,
                    baseDimensions);
            workingThreshold = thresholdAndGrade.index;
            workingGrade = thresholdAndGrade.weight;
        } else {
            workingThreshold = thresholder.getPrimaryThreshold();
            workingGrade = thresholder.getPrimaryGrade(score);
        }

        /*
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         */
        if (workingGrade <= 0) {
            result.setAnomalyGrade(0);
            result.setThreshold(workingThreshold);
            result.setInHighScoreRegion(false);
            thresholder.update(score, score, lastScore, result.transformMethod);
            if (shingleSize > 1) {
                // otherwise it will remain 0
                lastScore = score;
            }
            return result;
        }

        // the score is now high enough to be considered a candidate anomaly
        result.setInHighScoreRegion(true);

        /*
         * We now check if (1) we have another anomaly in the current shingle (2) have
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

        int gap = (int) (internalTimeStamp - lastAnomalyDescriptor.getInternalTimeStamp());

        // the forecast may not be reasonable with less data
        boolean reasonableForecast = result.isReasonableForecast();

        // note that the following is bypassed for shingleSize = 1 because it would not
        // make sense to connect the current evaluation with a previous value
        if (reasonableForecast && lastAnomalyDescriptor.getRCFPoint() != null && shingleSize > 1
                && lastAnomalyDescriptor.getExpectedRCFPoint() != null && gap > 0 && gap <= shingleSize) {
            double[] correctedPoint = applyBasicCorrector(point, gap, shingleSize, baseDimensions,
                    lastAnomalyDescriptor);
            double correctedScore = 0;
            if (result.forestMode != ForestMode.DISTANCE) {
                correctedScore = forest.getAnomalyScore(correctedPoint);
                workingGrade = thresholder.getThresholdAndGrade(correctedScore, result.transformMethod,
                        baseDimensions).weight;
            } else {
                correctedScore = forest.getSimpleDensity(correctedPoint).distances.getHighLowSum();
                workingGrade = thresholder.getPrimaryGrade(correctedScore);
            }
            // we know we are looking previous anomalies
            if (workingGrade == 0) {
                // fixing the past makes this anomaly go away; nothing to do but process the
                // score we will not change inHighScoreRegion however, because the score has
                // been larger
                result.setThreshold(workingThreshold);
                thresholder.update(score, correctedScore, lastScore, result.transformMethod);
                lastScore = correctedScore;
                result.setExpectedRCFPoint(correctedPoint);
                result.setAnomalyGrade(0);
                return result;
            }
        }

        /*
         * We now check the most egregious values seen in the current timestamp, as
         * determined by attribution. Those locations provide information about (a)
         * which attributes and (b) what the values should have been. However, those
         * calculations of imputation only make sense when sufficient observations are
         * available.
         */

        if (result.forestMode != ForestMode.DISTANCE) {
            attribution = forest.getAnomalyAttribution(point);
        }

        double[] newPoint = null;
        double newScore = score;
        DiVector newAttribution = null;

        /*
         * we now find the time slice, relative to the current time, which is indicative
         * of the high score. relativeIndex = 0 is current time. It is negative if the
         * most egregious attribution was due to the past values in the shingle
         */

        int relative = (gap >= shingleSize) ? -shingleSize : -gap;
        int index = (shingleSize == 1) ? 0 : maxContribution(attribution, baseDimensions, relative) + 1;
        boolean significant = true;
        if (reasonableForecast && shingleSize > 1) {
            startPosition = shingleSize * baseDimensions + (index - 1) * baseDimensions;
            newPoint = getExpectedPoint(attribution, startPosition, baseDimensions, point, forest);
            if (newPoint != null && result.getForestMode() != ForestMode.DISTANCE) {
                newAttribution = forest.getAnomalyAttribution(newPoint);
                newScore = newAttribution.getHighLowSum();
                // score is large, significantly over the threshold, or the change of a single
                // entry
                // causes a significant change in anomaly score
                // and no anomaly has not yet been reported on this shingle
                boolean significantScore = score > 1.5 || score > workingThreshold + 0.25
                        || (score > newScore + 0.25 && gap > shingleSize);
                // ignore late anomalies for larger shingleSizes unless the score
                // is considered signficant
                significant = (shingleSize > 4 && index + shingleSize / 2 > 0) || significantScore;
                if (significant) {
                    // time augmented mode will be improved later -- that would require extra
                    // information
                    int base = (result.forestMode == ForestMode.TIME_AUGMENTED) ? baseDimensions - 1 : baseDimensions;
                    significant = isSignificant(significantScore, point, newPoint, startPosition, base, result);
                }
            }
        }

        /*
         * if we are transitioning from low score to high score range (given by
         * inAnomaly) then we check if the new part of the input could have triggered
         * anomaly on its own That decision is vended by trigger() which extrapolates a
         * partial shingle.
         */

        if (significant && trigger(attribution, gap, baseDimensions, newAttribution, result.transformMethod,
                lastAnomalyDescriptor)) {
            result.setExpectedRCFPoint(newPoint);
            result.setAnomalyGrade(workingGrade);
            result.setThreshold(workingThreshold);
            thresholder.update(score, newScore, lastScore, result.transformMethod);
        } else {
            // we will force the threshold to rise
            result.setThreshold(score);
            thresholder.update(score, score, lastScore, result.transformMethod);
            if (shingleSize > 1) {
                lastScore = score;
            }
            result.setAnomalyGrade(0);
            return result;
        }

        if (shingleSize > 1) {
            lastScore = score;
        }

        result.setAttribution(attribution);
        result.setRelativeIndex(index);
        return result;
    }

    public void setZfactor(double factor) {
        thresholder.setZfactor(factor);
    }

    public void setLowerThreshold(double lower) {
        thresholder.setLowerThreshold(lower);
    }

    public void setThresholdPersistence(double persistence) {
        thresholder.setThresholdPersistence(persistence);
    }

    public void setInitialThreshold(double initial) {
        thresholder.setInitialThreshold(initial);
    }

    public void setIgnoreNearExpectedFromAbove(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            this.ignoreNearExpectedFromAbove = Arrays.copyOf(ignoreSimilarShift, ignoreSimilarShift.length);
        }
    }

    public void setIgnoreNearExpectedFromBelow(double[] ignoreSimilarShift) {
        if (ignoreSimilarShift != null) {
            this.ignoreNearExpectedFromBelow = Arrays.copyOf(ignoreSimilarShift, ignoreSimilarShift.length);
        }
    }

    public double[] getIgnoreNearExpectedFromAbove() {
        return ignoreNearExpectedFromAbove;
    }

    public double[] getIgnoreNearExpectedFromBelow() {
        return ignoreNearExpectedFromBelow;
    }

    public void setNumberOfAttributors(int numberOfAttributors) {
        this.numberOfAttributors = numberOfAttributors;
    }

    public int getNumberOfAttributors() {
        return numberOfAttributors;
    }

    public double getLastScore() {
        return lastScore;
    }

    public void setLastScore(double score) {
        lastScore = score;
    }
}
