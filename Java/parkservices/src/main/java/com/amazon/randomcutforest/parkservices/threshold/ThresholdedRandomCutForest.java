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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.returntypes.DiVector;

@Getter
@Setter
public class ThresholdedRandomCutForest {

    public static int MINIMUM_OBSERVATIONS_FOR_EXPECTED = 100;

    // a parameter that determines if the current potential anomaly is describing
    // the same anomaly
    // within the same shingle or across different time points
    protected double ignoreSimilarFactor = 0.3;

    // a different test for anomalies in the same shingle
    protected double triggerFactor = 3.0;

    // saved attribution of the last seen anomaly
    protected long lastAnomalyTimeStamp;

    // score of the last anomaly
    protected double lastAnomalyScore;

    // attribution of the last anomaly
    protected DiVector lastAnomalyAttribution;

    // last processed score
    protected double lastScore;

    // actual value of the last anomalous point (shingle)
    double[] lastAnomalyPoint;

    // if sufficient observations are present, what is the expected values
    // (for the top numberOfAttributor fields) for the last anomaly
    double[] lastExpectedPoint;

    // indicates that previous point was a candidate anomaly
    boolean previousIsPotentialAnomaly;

    // indicates if we are in a region where scores are high; this can be useful in
    // its own right
    boolean inHighScoreRegion;

    // flag that determines if we should dedup similar anomalies not in the same
    // shingle, for example an
    // anomaly, with the same pattern is repeated across more than a shingle
    protected boolean ignoreSimilar;

    // for anomaly description we would only look at these may top attributors
    // note that expected value is not well-defined when this number is greater than
    // 1
    int numberOfAttributors = 2;

    protected RandomCutForest forest;
    protected IThresholder thresholder;

    public ThresholdedRandomCutForest(RandomCutForest.Builder builder, double anomalyRate) {
        forest = builder.build();
        checkArgument(!forest.isInternalShinglingEnabled(), "Incorrect setting, not supported");
        thresholder = new BasicThresholder(anomalyRate);
        if (forest.getDimensions() / forest.getShingleSize() == 1) {
            thresholder.setLowerThreshold(1.1);
        }
    }

    public ThresholdedRandomCutForest(RandomCutForest forest, IThresholder thresholder) {
        this.forest = forest;
        this.thresholder = thresholder;
    }

    public AnomalyDescriptor process(double[] point) {
        AnomalyDescriptor result = getAnomalyDescription(point);
        forest.update(point);
        return result;
    }

    public RandomCutForest getForest() {
        return forest;
    }

    public IThresholder getThresholder() {
        return thresholder;
    }

    protected boolean useLastScore() {
        return lastScore > 0 && !previousIsPotentialAnomaly;
    }

    /**
     * The core update mechanism for thresholding, note that the score is used in
     * the primary statistic in thresholder (which maintains two) and the secondary
     * statistic is the score difference Since RandomCutForests are stochastic data
     * structures, scores from individual trees follow a trajectory not unlike
     * martingales. The differencing eliminates the effect or a run of high/low
     * scores.
     * 
     * @param score       typically the score produced by the forest
     * @param secondScore either the score or a corrected score which simulates
     *                    "what if the past anomalies were not present"
     * @param flag        a flag to indicate if the last point was potential anomaly
     */
    protected void update(double score, double secondScore, boolean flag) {
        if (useLastScore()) {
            thresholder.update(score, secondScore - lastScore);
        }
        lastScore = score;
        previousIsPotentialAnomaly = flag;
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
    private int maxContribution(DiVector diVector, int baseDimension, int startIndex) {
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
     * given an attribution vector finds the top attributors corresponding to the
     * shingle entry marked by position
     *
     * @param diVector      attribution of the current point
     * @param position      the specific time slice we are focusing on, can be the
     *                      most recent or a previous one if we were analyzing past
     *                      anomalies (but still in the shingle)
     * @param baseDimension number of variables/attributes in original data
     * @param max_number    the number of top attributors we are seeking; this
     *                      should not be more than 2 or 3, it can be exceedingly
     *                      hard to reason about more
     * @return the specific attribute locations within the shingle which has the
     *         largest attribution in [position, position+baseDimension-1]
     */
    private int[] largestFeatures(DiVector diVector, int position, int baseDimension, int max_number) {
        if (baseDimension == 1) {
            return new int[] { position };
        }
        double sum = 0;
        double[] values = new double[baseDimension];
        for (int i = 0; i < baseDimension; i++) {
            sum += values[i] = diVector.getHighLowSum(i + position);
        }
        Arrays.sort(values);
        int pick = Math.min(max_number, baseDimension);
        double cutoff = values[baseDimension - pick];
        // we will now throw away top attributors which are insignificant (10%) of the
        // next value
        while (pick > 1) {
            if (values[baseDimension - pick] < 10 * values[baseDimension - pick + 1]) {
                --pick;
                cutoff = values[baseDimension - pick];
            }
        }
        int[] answer = new int[pick];
        int count = 0;
        for (int i = 0; i < baseDimension; i++) {
            if (diVector.getHighLowSum(i + position) >= cutoff && diVector.getHighLowSum(i + position) > sum * 0.1) {
                answer[count++] = position + i;
            }
        }
        return Arrays.copyOf(answer, count);
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
    protected boolean trigger(DiVector candidate, int gap, int baseDimension, DiVector ideal) {
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
                return thresholder.getAnomalyGrade(remainder * dimensions / difference, triggerFactor) > 0;
            } else {
                double differentialRemainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i])
                            + Math.abs(candidate.high[i] - ideal.high[i]);
                }
                return (differentialRemainder > ignoreSimilarFactor * lastAnomalyScore) && thresholder
                        .getAnomalyGrade(differentialRemainder * dimensions / difference, triggerFactor) > 0;
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
            return (sum > ignoreSimilarFactor * lastScore);
        }
    }

    /**
     * core routine which collates the information about the most recent point
     * 
     * @param point input (shingled) point
     * @return description containing scores, grade, confidence, expected values,
     *         attribution etc.
     */
    protected AnomalyDescriptor getAnomalyDescription(double[] point) {
        AnomalyDescriptor result = new AnomalyDescriptor();
        DiVector attribution = forest.getAnomalyAttribution(point);
        double score = attribution.getHighLowSum();
        result.setRcfScore(score);
        long timeStamp = forest.getTotalUpdates();
        result.setTimeStamp(timeStamp);
        result.setForestSize(forest.getNumberOfTrees());
        result.setAttribution(attribution);
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        double[] currentValues = new double[baseDimensions];
        int startPosition = (shingleSize - 1) * baseDimensions;
        System.arraycopy(point, startPosition, currentValues, 0, baseDimensions);
        result.setCurrentValues(currentValues);

        if (timeStamp == 332) {
            System.out.println("HA");
        }
        // the forecast may not be reasonable with less data
        boolean reasonableForecast = (timeStamp > MINIMUM_OBSERVATIONS_FOR_EXPECTED)
                && (shingleSize * baseDimensions >= 4);
        /**
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         */
        if (thresholder.getAnomalyGrade(score) == 0) {
            result.setAnomalyGrade(0);
            inHighScoreRegion = false;
            result.setInHighScoreRegion(inHighScoreRegion);
            update(score, score, false);
            return result;
        } else {
            inHighScoreRegion = true;
        }

        // the score is now high enough to be considered an anomaly
        result.setInHighScoreRegion(inHighScoreRegion);

        /**
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

        int gap = (int) (timeStamp - lastAnomalyTimeStamp);

        if (reasonableForecast && lastAnomalyPoint != null && lastExpectedPoint != null && gap < shingleSize) {
            double[] correctedPoint = Arrays.copyOf(point, point.length);
            if (point.length - gap * baseDimensions >= 0)
                System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                        point.length - gap * baseDimensions);
            double correctedScore = forest.getAnomalyScore(correctedPoint);
            if (thresholder.getAnomalyGrade(correctedScore) == 0) {
                // fixing the past makes this anomaly go away; nothing to do but process the
                // score
                // we will not change inAnomaly however, because the score has been larger
                update(score, correctedScore, false);
                result.setAnomalyGrade(0);
                return result;
            }
        }

        /**
         * We now check the most egregious values seen in the current timestamp, as
         * determined by attribution. Those locations provide information about (a)
         * which attributes and (b) what the values should have been. However, those
         * calculations of imputation only make sense when sufficient observations are
         * available.
         */

        double[] newPoint = null;
        double newScore = 0;
        DiVector newAttribution = null;
        if (reasonableForecast) {
            int[] likelyMissingIndices = largestFeatures(attribution, startPosition, baseDimensions,
                    numberOfAttributors);
            newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
            newAttribution = forest.getAnomalyAttribution(newPoint);
            newScore = forest.getAnomalyScore(newPoint);
        }

        /**
         * we now find the time slice, relative to the current time, which is indicative
         * of the high score. relativeIndex = 0 is current time. It is negative if the
         * most egregious attribution was due to the past values in the shingle
         */

        result.setRelativeIndex(maxContribution(attribution, baseDimensions, -shingleSize) + 1);

        /**
         * if we are transitioning from low score to high score range (given by
         * inAnomaly) then we check if the new part of the input could have triggered
         * anomaly on its own That decision is vended by trigger() which extrapolates a
         * partial shingle
         */
        if (!inHighScoreRegion && trigger(attribution, gap, baseDimensions, null)) {
            result.setAnomalyGrade(thresholder.getAnomalyGrade(score));
            lastAnomalyScore = score;
            inHighScoreRegion = true;
            result.setStartOfAnomaly(true);
            lastAnomalyAttribution = new DiVector(attribution);
            lastAnomalyTimeStamp = timeStamp;
            lastAnomalyPoint = Arrays.copyOf(point, point.length);
            update(score, score, true);
        } else {
            /**
             * we again check if the new input produces an anomaly/not on its own
             */
            if (trigger(attribution, gap, baseDimensions, newAttribution) && score > newScore) {
                result.setAnomalyGrade(thresholder.getAnomalyGrade(score));
                lastAnomalyScore = score;
                lastAnomalyAttribution = new DiVector(attribution);
                lastAnomalyTimeStamp = timeStamp;
                lastAnomalyPoint = Arrays.copyOf(point, point.length);
                update(score, score, true);
            } else {
                // not changing inAnomaly
                result.setAnomalyGrade(0);
                update(score, score, false);
            }
        }

        if (result.getAnomalyGrade() > 0) {
            result.setExpectedValuesPresent(reasonableForecast);
            if (result.getRelativeIndex() < 0 && result.isStartOfAnomaly()) {
                // anomaly in the past and detected late; repositioning the computation
                // index 0 is current time
                startPosition = attribution.getDimensions() + (result.getRelativeIndex() - 1) * baseDimensions;
                if (reasonableForecast) {
                    int[] missingIndices = largestFeatures(attribution, startPosition, baseDimensions,
                            numberOfAttributors);
                    newPoint = forest.imputeMissingValues(point, missingIndices.length, missingIndices);
                    double[] oldValues = new double[baseDimensions];
                    System.arraycopy(point, startPosition, oldValues, 0, baseDimensions);
                    result.setOldValues(oldValues);
                }
            }
            if (reasonableForecast) {
                double[] values = new double[baseDimensions];
                System.arraycopy(newPoint, startPosition, values, 0, baseDimensions);
                result.setExpectedValues(0, values, 1.0);
                lastExpectedPoint = Arrays.copyOf(newPoint, newPoint.length);
            } else {
                lastExpectedPoint = null;
            }

            double[] flattenedAttribution = new double[baseDimensions];
            for (int i = 0; i < baseDimensions; i++) {
                flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
            }
            result.setFlattenedAttribution(flattenedAttribution);
        }
        return result;
    }

    public DiVector getLastAnomalyAttribution() {
        return (lastAnomalyAttribution == null) ? null : new DiVector(lastAnomalyAttribution);
    }

    public void setLastAnomalyAttribution(DiVector diVector) {
        lastAnomalyAttribution = (diVector == null) ? null : new DiVector(diVector);
    }

    public double[] getLastAnomalyPoint() {
        return copyIfNotnull(lastAnomalyPoint);
    }

    public double[] getLastExpectedPoint() {
        return copyIfNotnull(lastExpectedPoint);
    }

    public void setLastAnomalyPoint(double[] point) {
        lastAnomalyPoint = copyIfNotnull(point);
    }

    public void setLastExpectedPoint(double[] point) {
        lastExpectedPoint = copyIfNotnull(point);
    }

    private double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

}
