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
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_COMPACT;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INITIAL_ACCEPT_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INTERNAL_SHINGLING_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_NUMBER_OF_TREES;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PARALLEL_EXECUTION_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PRECISION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static com.amazon.randomcutforest.config.FillIn.FIXEDVALUES;
import static com.amazon.randomcutforest.config.FillIn.PREVIOUS;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.FillIn;
import com.amazon.randomcutforest.config.Mode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time. The class is ideally
 * supposed to be used in one of three modes;
 *
 * Sparse : this corresponds to sequence of events and time stamps are augmented
 * to the incoming point. ShingleSize 4 or higher is preferable; lowering the
 * shingleSize may lose quality of anomalies and explanations. Internal
 * Shingling shoule be used.
 *
 * Bursty : this corresponds to intermittent data and again shinglesize of 4 or
 * more is preferable missing value can be filled with 0's or specified values
 * (a vector of same length as the input point specifying coordinate values.
 * Choosing to store the imputed intermediate values in RCF is an option.
 *
 * Moderately continuous: missing values are filled in via previous values or
 * using impute; the latter would use a minimum number of observations, and
 * would fill in via previous till that point is reached. ShingleSize is
 * recommended to be 4. Choosing to store the imputed intermediate values in RCF
 * is an option.
 *
 * Custom: change parameters themselves
 *
 */
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

    // for anomaly description we would only look at these many top attributors
    // AExpected value is not well-defined when this number is greater than 1
    // that being said there is no formal restriction other than the fact that the
    // answers would be error prone as this parameter is raised.
    int numberOfAttributors = 2;

    // this parameter is used in imputing missing values in the input
    int valuesSeen = 0;

    // the input corresponds to timestamp data and this statistic helps align input
    Deviation timeStampDeviation;

    // can be used to normalize data
    Deviation[] deviationList;

    // for possible normalization
    boolean normalizeValues = false;

    // recording the last seen timestamp
    long[] previousTimeStamps;

    // normalize time difference;
    boolean normalizeTime = false;

    // fraction of data that should re actual input
    double useImputedFraction = 0.5;

    // number of imputed values in stored shingle
    int numberOfImputed;

    // particular strategy for impute
    FillIn fillIn = PREVIOUS;

    // mode of operation
    Mode mode = Mode.STANDARD;

    // for FILL_VALUES
    double[] defaultFill;

    // last point
    double[] lastShingledPoint;

    // last shingled values (without normalization/change or augmentation by time)
    double[] lastShingledInput;

    // sequence number to stop normalization at
    int stopNormalization = 100;

    // used in normalization
    double clipFactor = 5.0;

    protected RandomCutForest forest;
    protected IThresholder thresholder;

    public ThresholdedRandomCutForest(Builder<?> builder) {
        int inputLength;
        mode = builder.mode;
        previousTimeStamps = new long[builder.shingleSize];
        lastShingledInput = new double[builder.dimensions];
        if (builder.mode == Mode.TIMEAUGMENTED) {
            inputLength = builder.dimensions / builder.shingleSize;
            builder.dimensions += builder.shingleSize;
            normalizeTime = builder.normalizeTime;
            // force internal shingling for this option
            builder.internalShinglingEnabled = Optional.of(true);
        } else if (builder.mode == Mode.STREAMINGIMPUTE) {
            inputLength = builder.dimensions / builder.shingleSize;
            this.fillIn = builder.fillin;
            numberOfImputed = builder.shingleSize;
            builder.internalShinglingEnabled = Optional.of(false);
            if (this.fillIn == FIXEDVALUES) {
                int baseDimension = builder.dimensions / builder.shingleSize;
                // shingling will be performed in this layer and not in forest
                // so that we control admittance of imputed shingles
                checkArgument(builder.fillValues.length == baseDimension,
                        " the number of values should match the shingled input");
                this.defaultFill = Arrays.copyOf(builder.fillValues, builder.fillValues.length);
            }
            this.useImputedFraction = builder.useImputedFraction.orElse(0.5);
        } else {
            boolean smallInput = builder.internalShinglingEnabled.orElse(DEFAULT_INTERNAL_SHINGLING_ENABLED);
            inputLength = (smallInput) ? builder.dimensions / builder.shingleSize : builder.dimensions;
        }
        // we store the full shingled point, even though the forest may (but also may
        // not) store this
        lastShingledPoint = new double[builder.dimensions];
        double discount = builder.timeDecay
                .orElse(1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY * builder.sampleSize));

        this.normalizeValues = builder.normalizeValues;
        if (this.normalizeValues) {
            deviationList = new Deviation[inputLength];
            for (int i = 0; i < inputLength; i++) {
                deviationList[i] = new Deviation(discount);
            }
        }
        timeStampDeviation = new Deviation(discount);
        forest = builder.buildForest();
        BasicThresholder basic = new BasicThresholder(builder.anomalyRate);
        if (forest.getDimensions() / forest.getShingleSize() == 1) {
            basic.setLowerThreshold(1.1);
            basic.setHorizon(0.75);
        }
        thresholder = basic;
    }

    public ThresholdedRandomCutForest(RandomCutForest forest, IThresholder thresholder, Deviation deviation,
            Deviation[] deviations) {
        this.forest = forest;
        this.thresholder = thresholder;
        this.timeStampDeviation = deviation;
        this.deviationList = deviations;
    }

    public AnomalyDescriptor process(double[] inputPoint, long timestamp) {
        double[] input = inputPoint;

        if (inputPoint.length == lastShingledInput.length) {
            lastShingledInput = Arrays.copyOf(inputPoint, inputPoint.length);
        } else {
            shiftLeft(lastShingledInput, inputPoint.length);
            copyAtEnd(lastShingledInput, inputPoint);
        }
        if (normalizeValues) {
            checkArgument(inputPoint.length == deviationList.length, " mismatch in length");
            input = new double[inputPoint.length];
            for (int i = 0; i < inputPoint.length; i++) {
                input[i] = (valuesSeen <= 1) ? 0 : map(inputPoint[i], normalizeValues, deviationList[i]);
            }
        }

        double[] scaledInput = input;

        int dimension = forest.getDimensions();
        int shingleSize = forest.getShingleSize();
        int baseDimension = dimension / shingleSize;
        if (mode == Mode.TIMEAUGMENTED) {
            // augment with time difference
            // this works for both shingleSize = 1 and (internalShingling && shingleSIze >
            // 1)
            scaledInput = new double[inputPoint.length + 1];
            System.arraycopy(input, 0, scaledInput, 0, input.length);
            scaledInput[input.length] = (valuesSeen <= 1) ? 0
                    : map(timestamp - previousTimeStamps[shingleSize - 1], normalizeTime, timeStampDeviation);
        } else if (mode == Mode.STREAMINGIMPUTE) {
            if (valuesSeen > 1) {
                checkArgument(timeStampDeviation.getMean() > 0, " incorrect timestamps for imputation");
                checkArgument(timestamp > previousTimeStamps[shingleSize - 1], "incorrect order of time");
                int gap = (int) Math
                        .floor((timestamp - previousTimeStamps[shingleSize - 1]) / timeStampDeviation.getMean());
                if (gap >= 1.5) {
                    checkArgument(inputPoint.length == baseDimension, "error in length");
                    for (int i = 0; i < gap - 1; i++) {
                        double[] newPart = extractNew(fillIn, baseDimension, lastShingledPoint);
                        shiftLeft(lastShingledPoint, baseDimension);
                        copyAtEnd(lastShingledPoint, newPart);
                        numberOfImputed++;
                        if (numberOfImputed < useImputedFraction * forest.getShingleSize()) {
                            forest.update(lastShingledPoint);
                        }
                    }
                }
            }
            shiftLeft(lastShingledPoint, baseDimension);
            copyAtEnd(lastShingledPoint, input);
            scaledInput = Arrays.copyOf(lastShingledPoint, dimension);
            numberOfImputed--;
        }
        // the following handles both external and internal shingling
        double[] point = forest.transformToShingledPoint(scaledInput);

        AnomalyDescriptor result = getAnomalyDescription(point, timestamp, inputPoint);
        if (valuesSeen > 0) {
            timeStampDeviation.update(timestamp - previousTimeStamps[shingleSize - 1]);
        }
        if (normalizeValues) {
            if (valuesSeen < stopNormalization) {
                for (int i = 0; i < inputPoint.length; i++) {
                    deviationList[i].update(inputPoint[i]);
                }
            }
        }
        for (int i = 0; i < shingleSize - 1; i++) {
            previousTimeStamps[i] = previousTimeStamps[i + 1];
        }
        previousTimeStamps[shingleSize - 1] = timestamp;
        ++valuesSeen;
        if (mode != Mode.STREAMINGIMPUTE || numberOfImputed < useImputedFraction * forest.getShingleSize()) {
            forest.update(scaledInput);
        }
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
    protected boolean trigger(DiVector candidate, int gap, int baseDimension, DiVector ideal,
            boolean previousIsPotentialAnomaly) {
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
    protected AnomalyDescriptor getAnomalyDescription(double[] point, long inputTimeStamp, double[] inputPoint) {
        AnomalyDescriptor result = new AnomalyDescriptor();
        DiVector attribution = forest.getAnomalyAttribution(point);
        double score = attribution.getHighLowSum();
        result.setRcfScore(score);
        long internalTimeStamp = forest.getTotalUpdates();
        result.setNumberOfUpdates(internalTimeStamp);
        result.setTimeStamp(inputTimeStamp);
        result.setForestSize(forest.getNumberOfTrees());
        result.setAttribution(attribution);
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;
        int adjustTime = (mode == Mode.TIMEAUGMENTED) ? 1 : 0;

        if (score > 0) {
            double[] currentValues = Arrays.copyOf(inputPoint, inputPoint.length);
            result.setCurrentValues(currentValues);
        }
        // the forecast may not be reasonable with less data
        boolean reasonableForecast = (internalTimeStamp > MINIMUM_OBSERVATIONS_FOR_EXPECTED)
                && (shingleSize * baseDimensions >= 4);
        /**
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         */
        if (thresholder.getAnomalyGrade(score, previousIsPotentialAnomaly) == 0) {
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

        int gap = (int) (internalTimeStamp - lastAnomalyTimeStamp);

        if (reasonableForecast && lastAnomalyPoint != null && lastExpectedPoint != null && gap < shingleSize) {
            double[] correctedPoint = Arrays.copyOf(point, point.length);
            if (point.length - gap * baseDimensions >= 0)
                System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                        point.length - gap * baseDimensions);
            double correctedScore = forest.getAnomalyScore(correctedPoint);
            // we know we are looking previous anomalies
            if (thresholder.getAnomalyGrade(correctedScore, true) == 0) {
                // fixing the past makes this anomaly go away; nothing to do but process the
                // score
                // we will not change inHighScoreRegion however, because the score has been
                // larger
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
        double newScore = score;
        DiVector newAttribution = null;
        int[] likelyMissingIndices;
        if (reasonableForecast) {
            likelyMissingIndices = largestFeatures(attribution, startPosition, baseDimensions, numberOfAttributors);
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

        if (!previousIsPotentialAnomaly && trigger(attribution, gap, baseDimensions, null, false)) {
            result.setAnomalyGrade(thresholder.getAnomalyGrade(score, false));
            lastAnomalyScore = score;
            inHighScoreRegion = true;
            result.setStartOfAnomaly(true);
            lastAnomalyAttribution = new DiVector(attribution);
            lastAnomalyTimeStamp = internalTimeStamp;
            lastAnomalyPoint = Arrays.copyOf(point, point.length);
            update(score, score, true);
        } else {
            /**
             * we again check if the new input produces an anomaly/not on its own
             */
            if (trigger(attribution, gap, baseDimensions, newAttribution, previousIsPotentialAnomaly)
                    && score > newScore) {
                result.setAnomalyGrade(thresholder.getAnomalyGrade(score, previousIsPotentialAnomaly));
                lastAnomalyScore = score;
                lastAnomalyAttribution = new DiVector(attribution);
                lastAnomalyTimeStamp = internalTimeStamp;
                lastAnomalyPoint = Arrays.copyOf(point, point.length);
                update(score, newScore, true);
            } else {
                // not changing inAnomaly
                result.setAnomalyGrade(0);
                update(score, newScore, true);
            }
        }

        if (result.getAnomalyGrade() > 0) {
            result.setExpectedValuesPresent(reasonableForecast);
            if (result.getRelativeIndex() < 0 && result.isStartOfAnomaly()) {
                // anomaly in the past and detected late; repositioning the computation
                // index 0 is current time
                startPosition = attribution.getDimensions() + (result.getRelativeIndex() - 1) * baseDimensions;
                if (reasonableForecast) {
                    likelyMissingIndices = largestFeatures(attribution, startPosition, baseDimensions,
                            numberOfAttributors);
                    newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
                    double[] oldValues = new double[baseDimensions - adjustTime];
                    for (int i = 0; i < baseDimensions - adjustTime; i++) {
                        oldValues[i] = inverseMap(point[startPosition + i], normalizeValues, deviationList, i);
                    }
                    result.setOldValues(oldValues);
                    result.setOldTimeStamp(previousTimeStamps[shingleSize - 1 + result.getRelativeIndex()]);
                    if (mode == Mode.TIMEAUGMENTED) {
                        result.setExpectedTimeStamp(inverseMapTime(newPoint[startPosition + baseDimensions - 1],
                                result.getRelativeIndex(), normalizeTime));
                    }
                }
            } else {
                if (reasonableForecast) {
                    if (mode == Mode.TIMEAUGMENTED) {
                        result.setExpectedTimeStamp(
                                inverseMapTime(newPoint[startPosition + baseDimensions - 1], 0, normalizeTime));
                    }
                }
            }
            if (reasonableForecast) {
                double[] values = new double[baseDimensions - adjustTime];
                for (int i = 0; i < baseDimensions - adjustTime; i++) {
                    if (newPoint[startPosition + i] != point[startPosition + i]) {
                        values[i] = inverseMap(newPoint[startPosition + i], normalizeValues, deviationList, i);
                    } else {
                        values[i] = inputPoint[i];
                    }
                }
                result.setExpectedValues(0, values, 1.0);
                lastExpectedPoint = Arrays.copyOf(newPoint, newPoint.length);
            } else {
                lastExpectedPoint = null;
            }

            double[] flattenedAttribution = new double[baseDimensions - adjustTime];
            for (int i = 0; i < baseDimensions - adjustTime; i++) {
                flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
            }
            result.setFlattenedAttribution(flattenedAttribution);
            if (mode == Mode.TIMEAUGMENTED) {
                result.setTimeAttribution(attribution.getHighLowSum(startPosition + baseDimensions - 1));
            }
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

    public Deviation getTimeStampDeviation() {
        return timeStampDeviation;
    }

    public void setLastExpectedPoint(double[] point) {
        lastExpectedPoint = copyIfNotnull(point);
    }

    private double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    /**
     * maps a value shifted to the current mean or to a relative space
     * 
     * @param value     input value of dimension
     * @param normalize flag that determines relative or shift
     * @param deviation statistic
     * @return the normalized value
     */
    double map(double value, boolean normalize, Deviation deviation) {
        if (normalize) {
            if (value - deviation.getMean() >= 2 * clipFactor * deviation.getDeviation()) {
                return clipFactor;
            }
            if (value - deviation.getMean() < -2 * clipFactor * deviation.getDeviation()) {
                return -clipFactor;
            } else {
                // deviation cannot be 0
                return (value - deviation.getMean()) / (2 * deviation.getDeviation());
            }
        } else {
            return value - deviation.getMean();
        }
    }

    /**
     * inverse map for constructing values in the input space
     * 
     * @param gap        internal value
     * @param normalize  flag indicating relative (true) or shift (false)
     * @param deviations statistic array
     * @param index      index of statistic
     * @return inverse mapped value
     */
    double inverseMap(double gap, boolean normalize, Deviation[] deviations, int index) {
        if (normalize) {
            return deviations[index].getMean() + 2 * gap * deviations[index].getDeviation();
        } else {
            return gap;
        }
    }

    /**
     * maps the time back. The returned value is an approximation for
     * relativePoisition < 0 which corresponds to an anomaly in the past. Since the
     * state of the statistic is now changed based on more recent values
     * 
     * @param gap              estimated value
     * @param relativePosition how far back in the shingle
     * @param normalize        the flag that determines normalization
     * @return transform of the time value to original input space
     */
    long inverseMapTime(double gap, int relativePosition, boolean normalize) {
        checkArgument(forest.getShingleSize() + relativePosition >= 0, " error");
        int shingleSize = forest.getShingleSize();
        if (normalize) {
            return (long) Math.floor(previousTimeStamps[shingleSize - 1 + relativePosition]
                    + timeStampDeviation.getMean() + 2 * gap * timeStampDeviation.getDeviation());
        } else {
            return (long) Math
                    .floor(previousTimeStamps[shingleSize - 1 + relativePosition] + gap + timeStampDeviation.getMean());
        }
    }

    /**
     * determines the next point to fed to the model for a missing value
     * 
     * @param fillin            strategy of imputation
     * @param baseDimension     number of values to imput
     * @param lastShingledPoint last full entry seen by model
     * @return an array of baseDimension values corresponding to plausible next
     *         input
     */

    double[] extractNew(FillIn fillin, int baseDimension, double[] lastShingledPoint) {
        double[] result = new double[baseDimension];
        if (fillin == FillIn.ZERO) {
            return result;
        }
        if (fillin == FIXEDVALUES) {
            System.arraycopy(defaultFill, 0, result, 0, baseDimension);
            return result;
        }
        int dimension = forest.getDimensions();
        if (fillin == PREVIOUS || forest.getTotalUpdates() < MINIMUM_OBSERVATIONS_FOR_EXPECTED && dimension >= 4
                && baseDimension <= 2) {
            System.arraycopy(lastShingledPoint, dimension - baseDimension, result, 0, baseDimension);
            return result;
        }
        int[] positions = new int[baseDimension];
        double[] temp = Arrays.copyOf(lastShingledPoint, lastShingledPoint.length);
        shiftLeft(temp, baseDimension);
        for (int y = 0; y < baseDimension; y++) {
            positions[y] = dimension - baseDimension + y;
        }
        double[] newPoint = forest.imputeMissingValues(temp, baseDimension, positions);
        System.arraycopy(newPoint, dimension - baseDimension, result, 0, baseDimension);
        return result;
    }

    void shiftLeft(double[] array, int baseDimension) {
        for (int i = 0; i < array.length - baseDimension; i++) {
            array[i] = array[i + baseDimension];
        }
    }

    long[] getPreviousTimeStamps() {
        return (previousTimeStamps == null) ? null : Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
    }

    void setPreviousTimeStamps(long[] values) {
        previousTimeStamps = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    void copyAtEnd(double[] array, double[] small) {
        checkArgument(array.length > small.length, " incorrect operation ");
        for (int i = 0; i < small.length; i++) {
            array[array.length - small.length + i] = small[i];
        }
    }

    public double[] getLastShingledPoint() {
        return copyIfNotnull(lastShingledPoint);
    }

    public void setLastShingledPoint(double[] point) {
        lastShingledPoint = copyIfNotnull(point);
    }

    public double[] getLastShingledInput() {
        return copyIfNotnull(lastShingledInput);
    }

    public void setLastShingledInput(double[] point) {
        lastShingledInput = copyIfNotnull(point);
    }

    public double[] getDefaultFill() {
        return copyIfNotnull(defaultFill);
    }

    public void setDefaultFill(double[] values) {
        defaultFill = copyIfNotnull(values);
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        protected int dimensions;
        protected int sampleSize = DEFAULT_SAMPLE_SIZE;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected int numberOfTrees = DEFAULT_NUMBER_OF_TREES;
        protected Optional<Double> timeDecay = Optional.empty();
        protected Optional<Long> randomSeed = Optional.empty();
        protected boolean compact = DEFAULT_COMPACT;
        protected boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        protected boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        protected boolean parallelExecutionEnabled = DEFAULT_PARALLEL_EXECUTION_ENABLED;
        protected Optional<Integer> threadPoolSize = Optional.empty();
        protected Precision precision = DEFAULT_PRECISION;
        protected double boundingBoxCacheFraction = DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected int shingleSize = DEFAULT_SHINGLE_SIZE;
        protected Optional<Boolean> internalShinglingEnabled = Optional.empty();
        protected double initialAcceptFraction = DEFAULT_INITIAL_ACCEPT_FRACTION;
        protected double anomalyRate = 0.01;
        protected FillIn fillin = PREVIOUS;
        protected Mode mode = Mode.STANDARD;
        protected boolean normalizeTime = false;
        protected boolean normalizeValues = false;
        protected double[] fillValues = null;
        protected Optional<Double> useImputedFraction = Optional.empty();

        void validate() {
            if (mode == Mode.TIMEAUGMENTED) {
                if (internalShinglingEnabled.isPresent()) {
                    checkArgument(shingleSize == 1 || internalShinglingEnabled.get(),
                            " shingle size has to be 1 or " + "internal shingling must turned on");
                } else {
                    internalShinglingEnabled = Optional.of(true);
                }
                if (useImputedFraction.isPresent()) {
                    throw new IllegalArgumentException(" imputation infeasible");
                }
            } else if (mode == Mode.STREAMINGIMPUTE) {
                checkArgument(shingleSize > 1, "imputation with shingle size 1 is not meaningful");
                internalShinglingEnabled.ifPresent(x -> checkArgument(x, " nothing to impute for full shingles"));
            } else if (!internalShinglingEnabled.isPresent()) {
                internalShinglingEnabled = Optional.of(false);
                if (useImputedFraction.isPresent()) {
                    throw new IllegalArgumentException(" imputation infeasible");
                }
            }
        }

        public ThresholdedRandomCutForest build() {
            validate();
            return new ThresholdedRandomCutForest(this);
        }

        protected RandomCutForest buildForest() {
            RandomCutForest.Builder<?> builder = new RandomCutForest.Builder<>().dimensions(dimensions)
                    .sampleSize(sampleSize).numberOfTrees(numberOfTrees).compact(compact)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled).centerOfMassEnabled(centerOfMassEnabled)
                    .parallelExecutionEnabled(parallelExecutionEnabled).precision(precision)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).shingleSize(shingleSize)
                    .internalShinglingEnabled(internalShinglingEnabled.get())
                    .initialAcceptFraction(initialAcceptFraction);
            outputAfter.ifPresent(builder::outputAfter);
            timeDecay.ifPresent(builder::timeDecay);
            randomSeed.ifPresent(builder::randomSeed);
            threadPoolSize.ifPresent(builder::threadPoolSize);
            return builder.build();
        }

        public T dimensions(int dimensions) {
            this.dimensions = dimensions;
            return (T) this;
        }

        public T sampleSize(int sampleSize) {
            this.sampleSize = sampleSize;
            return (T) this;
        }

        public T outputAfter(int outputAfter) {
            this.outputAfter = Optional.of(outputAfter);
            return (T) this;
        }

        public T numberOfTrees(int numberOfTrees) {
            this.numberOfTrees = numberOfTrees;
            return (T) this;
        }

        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        public T timeDecay(double timeDecay) {
            this.timeDecay = Optional.of(timeDecay);
            return (T) this;
        }

        public T useImputedFraction(double fraction) {
            this.useImputedFraction = Optional.of(fraction);
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = Optional.of(randomSeed);
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public T parallelExecutionEnabled(boolean parallelExecutionEnabled) {
            this.parallelExecutionEnabled = parallelExecutionEnabled;
            return (T) this;
        }

        public T threadPoolSize(int threadPoolSize) {
            this.threadPoolSize = Optional.of(threadPoolSize);
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T compact(boolean compact) {
            this.compact = compact;
            return (T) this;
        }

        public T internalShinglingEnabled(boolean internalShinglingEnabled) {
            this.internalShinglingEnabled = Optional.of(internalShinglingEnabled);
            return (T) this;
        }

        public T precision(Precision precision) {
            this.precision = precision;
            return (T) this;
        }

        public T boundingBoxCacheFraction(double boundingBoxCacheFraction) {
            this.boundingBoxCacheFraction = boundingBoxCacheFraction;
            return (T) this;
        }

        public T initialAcceptFraction(double initialAcceptFraction) {
            this.initialAcceptFraction = initialAcceptFraction;
            return (T) this;
        }

        public Random getRandom() {
            // If a random seed was given, use it to create a new Random. Otherwise, call
            // the 0-argument constructor
            return randomSeed.map(Random::new).orElseGet(Random::new);
        }

        public T anomalyRate(double anomalyRate) {
            this.anomalyRate = anomalyRate;
            return (T) this;
        }

        public T fillIn(FillIn fillIn) {
            this.fillin = fillIn;
            return (T) this;
        }

        public T fillValues(double[] values) {
            // values cannot be a null
            this.fillValues = Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T normalizeTime(boolean normalizeTime) {
            this.normalizeTime = normalizeTime;
            return (T) this;
        }

        public T normalizeValues(boolean normalizeValues) {
            this.normalizeValues = normalizeValues;
            return (T) this;
        }

        public T setMode(Mode mode) {
            this.mode = mode;
            return (T) this;
        }

    }
}
