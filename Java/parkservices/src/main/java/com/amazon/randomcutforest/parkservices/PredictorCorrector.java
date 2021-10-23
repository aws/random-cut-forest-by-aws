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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
@Getter
@Setter
public class PredictorCorrector {

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 5;

    public static double DEFAULT_REPEAT_ANOMALY_Z_FACTOR = 3.5;

    public static double DEFAULT_IGNORE_SIMILAR_FACTOR = 0.3;

    public static boolean DEFAULT_IGNORE_SIMILAR = false;

    // a parameter that determines if the current potential anomaly is describing
    // the same anomaly
    // within the same shingle or across different time points
    protected double ignoreSimilarFactor = DEFAULT_IGNORE_SIMILAR_FACTOR;

    // a different test for anomalies in the same shingle
    protected double triggerFactor = DEFAULT_REPEAT_ANOMALY_Z_FACTOR;

    // flag that determines if we should dedup similar anomalies not in the same
    // shingle, for example an
    // anomaly, with the same pattern is repeated across more than a shingle
    protected boolean ignoreSimilar = DEFAULT_IGNORE_SIMILAR;

    // for anomaly description we would only look at these many top attributors
    // AExpected value is not well-defined when this number is greater than 1
    // that being said there is no formal restriction other than the fact that the
    // answers would be error prone as this parameter is raised.
    protected int numberOfAttributors = DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS;

    protected BasicThresholder thresholder;

    // for mappers
    public PredictorCorrector(BasicThresholder thresholder) {
        this.thresholder = thresholder;
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
     * @param candidate                  attribution of the current point in
     *                                   consideration
     * @param gap                        how long ago did the previous anomaly occur
     * @param baseDimension              number of input attributes/variables
     *                                   (before shingling)
     * @param ideal                      a form of expected attribution; can be null
     *                                   if there was no previous anomaly in the
     *                                   shingle
     * @param previousIsPotentialAnomaly is the previous point a potential anomaly
     * @return true/false if the residual (extrapolated) score would trigger anomaly
     *         designation
     */
    protected boolean trigger(DiVector candidate, int gap, int baseDimension, DiVector ideal,
            boolean previousIsPotentialAnomaly, IRCFComputeDescriptor lastAnomalyDescriptor) {
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
                return thresholder.getAnomalyGrade(remainder * dimensions / difference, previousIsPotentialAnomaly,
                        triggerFactor) > 0;
            } else {
                double differentialRemainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i])
                            + Math.abs(candidate.high[i] - ideal.high[i]);
                }
                return (differentialRemainder > ignoreSimilarFactor * lastAnomalyScore)
                        && thresholder.getAnomalyGrade(differentialRemainder * dimensions / difference,
                                previousIsPotentialAnomaly, triggerFactor) > 0;
            }
        } else {
            if (!ignoreSimilar) {
                return true;
            }
            double sum = 0;
            for (int i = 0; i < dimensions; i++) {
                sum += Math.abs(lastAnomalyAttribution.high[i] - candidate.high[i])
                        + Math.abs(lastAnomalyAttribution.low[i] - candidate.low[i]);
            }
            return (sum > ignoreSimilarFactor * lastAnomalyScore);
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
        double[] correctedPoint = Arrays.copyOf(point, point.length);
        double[] lastExpectedPoint = lastAnomalyDescriptor.getExpectedRCFPoint();
        double[] lastAnomalyPoint = lastAnomalyDescriptor.getRCFPoint();
        int lastRelativeIndex = lastAnomalyDescriptor.getRelativeIndex();
        if (gap < shingleSize) {
            System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                    point.length - gap * baseDimensions);
        }
        if (lastRelativeIndex == 0) { // is is possible to fix other cases, but is more complicated
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
     * the core of the predictor-corrector thresholding for shingled data points. It
     * uses a simple threshold provided by the basic thresholder. It first checks if
     * obvious effects of the present; and absent such, for repeated breaches, how
     * critical is the new current information
     * 
     * @param result                returns the augmented description
     * @param lastAnomalyDescriptor state of the computation for the last anomaly
     * @return the anomaly descriptor result (which has plausibly mutated)
     */
    protected AnomalyDescriptor detect(AnomalyDescriptor result, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {
        double[] point = result.getRCFPoint();
        if (point == null) {
            return result;
        }
        double score = forest.getAnomalyScore(point);
        result.setRCFScore(score);
        result.setRCFPoint(point);
        long internalTimeStamp = result.getInternalTimeStamp();

        if (score == 0) {
            return result;
        }

        int shingleSize = result.getShingleSize();
        int baseDimensions = result.getDimension() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;

        result.setThreshold(thresholder.threshold());

        boolean previousIsPotentialAnomaly = thresholder.isInPotentialAnomaly();

        /*
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         */
        if (thresholder.getAnomalyGrade(score, previousIsPotentialAnomaly) == 0) {
            result.setAnomalyGrade(0);
            // inHighScoreRegion = false;
            result.setInHighScoreRegion(false);
            thresholder.update(score, score, 0, false);
            return result;
        }

        // the score is now high enough to be considered an anomaly
        // inHighScoreRegion = true;
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

        if (reasonableForecast && lastAnomalyDescriptor.getRCFPoint() != null
                && lastAnomalyDescriptor.getExpectedRCFPoint() != null && gap > 0 && gap <= shingleSize) {
            double[] correctedPoint = applyBasicCorrector(point, gap, shingleSize, baseDimensions,
                    lastAnomalyDescriptor);
            double correctedScore = forest.getAnomalyScore(correctedPoint);
            // we know we are looking previous anomalies
            if (thresholder.getAnomalyGrade(correctedScore, true) == 0) {
                // fixing the past makes this anomaly go away; nothing to do but process the
                // score
                // we will not change inHighScoreRegion however, because the score has been
                // larger
                thresholder.update(score, correctedScore, 0, false);
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

        DiVector attribution = forest.getAnomalyAttribution(point);

        double[] newPoint = null;
        double newScore = score;
        DiVector newAttribution = null;

        /*
         * we now find the time slice, relative to the current time, which is indicative
         * of the high score. relativeIndex = 0 is current time. It is negative if the
         * most egregious attribution was due to the past values in the shingle
         */

        int index = maxContribution(attribution, baseDimensions, -shingleSize) + 1;

        /*
         * if we are transitioning from low score to high score range (given by
         * inAnomaly) then we check if the new part of the input could have triggered
         * anomaly on its own That decision is vended by trigger() which extrapolates a
         * partial shingle
         */

        if (!previousIsPotentialAnomaly
                && trigger(attribution, gap, baseDimensions, null, false, lastAnomalyDescriptor)) {
            result.setAnomalyGrade(thresholder.getAnomalyGrade(score, false));
            result.setStartOfAnomaly(true);
            thresholder.update(score, score, 0, true);
        } else {
            /*
             * we again check if the new input produces an anomaly/not on its own
             */
            if (reasonableForecast) {
                newPoint = getExpectedPoint(attribution, startPosition, baseDimensions, point, forest);
                if (newPoint != null) {
                    newAttribution = forest.getAnomalyAttribution(newPoint);
                    newScore = forest.getAnomalyScore(newPoint);
                    result.setExpectedRCFPoint(newPoint);
                }
            }

            if (trigger(attribution, gap, baseDimensions, newAttribution, previousIsPotentialAnomaly,
                    lastAnomalyDescriptor) && score > newScore) {
                result.setAnomalyGrade(thresholder.getAnomalyGrade(score, previousIsPotentialAnomaly));
                index = 0; // current point
                thresholder.update(score, newScore, 0, true);
            } else {
                // previousIsPotentialAnomaly is true now, but not calling it anomaly either
                thresholder.update(score, newScore, 0, true);
                result.setAnomalyGrade(0);
                return result;
            }
        }

        result.setAttribution(attribution);
        result.setRelativeIndex(index);
        if (reasonableForecast) {
            // anomaly in the past and detected late; repositioning the computation
            // index 0 is current time
            startPosition = shingleSize * baseDimensions + (result.getRelativeIndex() - 1) * baseDimensions;
            newPoint = getExpectedPoint(result.getAttribution(), startPosition, baseDimensions, point, forest);
        }
        result.setExpectedRCFPoint(newPoint);
        return result;
    }

    public void setZfactor(double factor) {
        thresholder.setZfactor(factor);
        triggerFactor = Math.max(factor, triggerFactor);
    }

    public void setLowerThreshold(double lower) {
        thresholder.setAbsoluteThreshold(lower);
    }

    public void setHorizon(double horizon) {
        thresholder.setHorizon(horizon);
    }

    public void setInitialThreshold(double initial) {
        thresholder.setInitialThreshold(initial);
    }

}
