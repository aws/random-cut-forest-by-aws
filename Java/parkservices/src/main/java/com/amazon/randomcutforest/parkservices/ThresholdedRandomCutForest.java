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
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_HORIZON;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_HORIZON_ONED;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD_NORMALIZED;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD_NORMALIZED_ONED;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD_ONED;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
@Getter
@Setter
public class ThresholdedRandomCutForest {

    public static int MINIMUM_OBSERVATIONS_FOR_EXPECTED = 100;

    public static double DEFAULT_USE_IMPUTED_FRACTION = 0.5;

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 2;

    public static double DEFAULT_REPEAT_ANOMALY_Z_FACTOR = 3.5;

    public static double DEFAULT_IGNORE_SIMILAR_FACTOR = 0.3;

    public static boolean DEFAULT_IGNORE_SIMILAR = false;

    // a parameter that determines if the current potential anomaly is describing
    // the same anomaly
    // within the same shingle or across different time points
    protected double ignoreSimilarFactor = DEFAULT_IGNORE_SIMILAR_FACTOR;

    // a different test for anomalies in the same shingle
    protected double triggerFactor = DEFAULT_REPEAT_ANOMALY_Z_FACTOR;

    // saved attribution of the last seen anomaly
    protected long lastAnomalyTimeStamp;

    // score of the last anomaly
    protected double lastAnomalyScore;

    // attribution of the last anomaly
    protected DiVector lastAnomalyAttribution;

    // last processed score
    protected double lastScore;

    // actual value of the last anomalous point (shingle)
    protected double[] lastAnomalyPoint;

    // location of the anomaly in the last anomalous shingle
    protected int lastRelativeIndex = 0;

    // if sufficient observations are present, what is the expected values
    // (for the top numberOfAttributor fields) for the last anomaly
    protected double[] lastExpectedPoint;

    // indicates that previous point was a candidate anomaly
    protected boolean previousIsPotentialAnomaly;

    // indicates if we are in a region where scores are high; this can be useful in
    // its own right
    protected boolean inHighScoreRegion;

    // flag that determines if we should dedup similar anomalies not in the same
    // shingle, for example an
    // anomaly, with the same pattern is repeated across more than a shingle
    protected boolean ignoreSimilar = DEFAULT_IGNORE_SIMILAR;

    // for anomaly description we would only look at these many top attributors
    // AExpected value is not well-defined when this number is greater than 1
    // that being said there is no formal restriction other than the fact that the
    // answers would be error prone as this parameter is raised.
    protected int numberOfAttributors = DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS;

    // forestMode of operation
    protected ForestMode forestMode = ForestMode.STANDARD;

    protected TransformMethod transformMethod = TransformMethod.NONE;

    protected RandomCutForest forest;
    protected BasicThresholder thresholder;
    protected Preprocessor preprocessor;

    public ThresholdedRandomCutForest(Builder<?> builder) {

        forestMode = builder.forestMode;
        transformMethod = builder.transformMethod;
        Preprocessor.Builder<?> preprocessorBuilder = Preprocessor.builder().shingleSize(builder.shingleSize)
                .transformMethod(builder.transformMethod).forestMode(builder.forestMode);

        if (builder.forestMode == ForestMode.TIME_AUGMENTED) {
            preprocessorBuilder.inputLength(builder.dimensions / builder.shingleSize);
            builder.dimensions += builder.shingleSize;
            preprocessorBuilder.normalizeTime(builder.normalizeTime);
            // force internal shingling for this option
            builder.internalShinglingEnabled = Optional.of(true);
        } else if (builder.forestMode == ForestMode.STREAMING_IMPUTE) {
            preprocessorBuilder.inputLength(builder.dimensions / builder.shingleSize);
            preprocessorBuilder.imputationMethod(builder.imputationMethod);
            if (builder.fillValues != null) {
                preprocessorBuilder.fillValues(builder.fillValues);
            }
            builder.internalShinglingEnabled = Optional.of(false);
            preprocessorBuilder.useImputedFraction(builder.useImputedFraction.orElse(0.5));
        } else {
            boolean smallInput = builder.internalShinglingEnabled.orElse(DEFAULT_INTERNAL_SHINGLING_ENABLED);
            preprocessorBuilder
                    .inputLength((smallInput) ? builder.dimensions / builder.shingleSize : builder.dimensions);
        }

        preprocessorBuilder.weights(builder.weights);
        preprocessorBuilder.timeDecay(
                builder.timeDecay.orElse(1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY * builder.sampleSize)));
        preprocessorBuilder.dimensions(builder.dimensions);
        preprocessorBuilder
                .stopNormalization(builder.stopNormalization.orElse(Preprocessor.DEFAULT_STOP_NORMALIZATION));
        preprocessorBuilder
                .startNormalization(builder.startNormalization.orElse(Preprocessor.DEFAULT_START_NORMALIZATION));
        forest = builder.buildForest();

        BasicThresholder basic = new BasicThresholder(builder.anomalyRate, builder.adjustThreshold);
        // multiple (not extremely well correlated) dimensions typically reduce scores
        // normalization reduces scores
        if (forest.getDimensions() / forest.getShingleSize() == 1) {
            if (builder.transformMethod != TransformMethod.NORMALIZE) {
                basic.setAbsoluteThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD_ONED));
            } else {
                basic.setAbsoluteThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD_NORMALIZED_ONED));
            }
            basic.setHorizon(builder.horizon.orElse(DEFAULT_HORIZON_ONED));
        } else {
            if (builder.transformMethod != TransformMethod.NORMALIZE) {
                basic.setAbsoluteThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD));
            } else {
                basic.setAbsoluteThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD_NORMALIZED));
            }
            basic.setHorizon(builder.horizon.orElse(DEFAULT_HORIZON));
        }
        thresholder = basic;
        preprocessor = preprocessorBuilder.build();
    }

    // for mappers
    public ThresholdedRandomCutForest(RandomCutForest forest, BasicThresholder thresholder, Preprocessor preprocessor) {
        this.forest = forest;
        this.thresholder = thresholder;
        this.preprocessor = preprocessor;
    }

    // for conversion from other thresholding models
    public ThresholdedRandomCutForest(RandomCutForest forest, double futureAnomalyRate, List<Double> values) {
        this.forest = forest;
        this.thresholder = new BasicThresholder(values, futureAnomalyRate);
        int dimensions = forest.getDimensions();
        int inputLength = (forest.isInternalShinglingEnabled()) ? dimensions / forest.getShingleSize()
                : forest.getDimensions();
        this.preprocessor = new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).dimensions(dimensions)
                .shingleSize(forest.getShingleSize()).inputLength(inputLength).build();
        preprocessor.setValuesSeen((int) forest.getTotalUpdates());
        preprocessor.getDataQuality().update(1.0);
    }

    /**
     * a single call that prepreprocesses data, compute score/grade and updates
     * state
     * 
     * @param inputPoint current input point
     * @param timestamp  time stamp of input
     * @return anomalydescriptor for the current input point
     */
    public AnomalyDescriptor process(double[] inputPoint, long timestamp) {

        double[] scaledInput = preprocessor.preProcess(inputPoint, timestamp, forest);

        if (scaledInput == null) {
            return new AnomalyDescriptor();
        }

        // the following handles both external and internal shingling
        double[] point = forest.transformToShingledPoint(scaledInput);

        // score anomalies
        AnomalyDescriptor result = getAnomalyDescription(point, timestamp, inputPoint);

        // update state
        preprocessor.postProcess(inputPoint, timestamp, forest, scaledInput);

        return result;

    }

    public RandomCutForest getForest() {
        return forest;
    }

    public BasicThresholder getThresholder() {
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
    protected int[] largestFeatures(DiVector diVector, int position, int baseDimension, int max_number) {
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

        // we will now throw away top attributors which are insignificant (10%) of the
        // next value
        while (pick > 1 && values[baseDimension - pick + 1] > 10 * values[baseDimension - pick]) {
            --pick;
        }
        double cutoff = values[baseDimension - pick];
        int[] answer = new int[pick];
        int count = 0;
        for (int i = 0; i < baseDimension; i++) {
            if (diVector.getHighLowSum(i + position) >= cutoff
                    && (count == 0 || diVector.getHighLowSum(i + position) > sum * 0.1)) {
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
     * once an anomaly is found, this records the expected values (provided there is
     * sufficient data to enable a meanigful value). It avoids calling the
     * imputation for a second time. It also provides the expected values for the
     * most likely place of anomaly (indicated by index) as opposed to likely values
     * for current input.
     * 
     * @param inputPoint the actual input point
     * @param point      the (potentially) transformed point for RCF
     * @param expected   the expected values computed previously in determining an
     *                   anomaly
     * @param result     the descriptor where information is stored
     * @param index      the relative position in the shingle where anomaly is
     *                   likely located
     */
    void addExpectedAndUpdateState(double[] inputPoint, double[] point, double[] expected, AnomalyDescriptor result,
            int index) {
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;
        int adjustTime = (forestMode == ForestMode.TIME_AUGMENTED) ? 1 : 0;
        int base = baseDimensions - adjustTime;
        double[] reference = inputPoint;
        double[] newPoint = expected;

        if (index < 0 && result.isStartOfAnomaly()) {
            // anomaly in the past and detected late; repositioning the computation
            // index 0 is current time
            startPosition = shingleSize * baseDimensions + (result.getRelativeIndex() - 1) * baseDimensions;

            int[] likelyMissingIndices = largestFeatures(result.getAttribution(), startPosition, baseDimensions,
                    numberOfAttributors);
            newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
            // preprocessor is not updated yet; note index is less than 0
            reference = preprocessor.getShingledInput(shingleSize + index);
            result.setOldValues(reference);
            result.setOldTimeStamp(preprocessor.getTimeStamp(shingleSize - 1 + index));
        }

        double[] differenceBase = preprocessor.getShingledInput(shingleSize - 1 + index);

        if (forestMode == ForestMode.TIME_AUGMENTED) {
            long expectedStamp = preprocessor.inverseMapTime(newPoint[startPosition + baseDimensions - 1],
                    result.getRelativeIndex());
            result.setExpectedTimeStamp(expectedStamp);
        }

        double[] values = new double[base];
        if (reference.length == base) {
            for (int i = 0; i < base; i++) {
                values[i] = (point[startPosition + i] == newPoint[startPosition + i]) ? reference[i]
                        : preprocessor.inverseTransform(newPoint[startPosition + i], i, differenceBase[i]);
            }
        } else {
            for (int i = 0; i < base; i++) {
                values[i] = (point[startPosition + i] == newPoint[startPosition + i]) ? reference[startPosition + i]
                        : preprocessor.inverseTransform(newPoint[startPosition + i], i, differenceBase[i]);
            }
        }
        result.setExpectedValues(0, values, 1.0);
        lastExpectedPoint = Arrays.copyOf(newPoint, newPoint.length);
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
    double applyBasicCorrector(double[] point, int gap, int shingleSize, int baseDimensions) {
        double[] correctedPoint = Arrays.copyOf(point, point.length);
        if (gap < shingleSize) {
            System.arraycopy(lastExpectedPoint, gap * baseDimensions, correctedPoint, 0,
                    point.length - gap * baseDimensions);
        }
        if (lastRelativeIndex == 0) { // is is possible to fix other cases, but is more complicated
            if (transformMethod == TransformMethod.DIFFERENCE
                    || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
                // this works for time augmentation since time is always differenced
                for (int y = 0; y < baseDimensions; y++) {
                    correctedPoint[point.length - gap * baseDimensions
                            + y] += lastAnomalyPoint[point.length - baseDimensions + y]
                                    - lastExpectedPoint[point.length - baseDimensions + y];
                }
            } else if (forestMode == ForestMode.TIME_AUGMENTED) {
                // need to fix the effect of the timestamp on the next stamp due to differencing
                // in case of normalization there is an issue is discount is set too high (but
                // not for the default)
                correctedPoint[point.length - (gap - 1) * baseDimensions - 1] += lastAnomalyPoint[point.length - 1]
                        - lastExpectedPoint[point.length - 1];
            }
        }
        return forest.getAnomalyScore(correctedPoint);
    }

    /**
     * adds the attribution values for the current time slice in the shingle; as
     * opposed to aggregate information across the entire shingle
     * 
     * @param baseDimensions number of entries in each shingle
     * @param adjustTime     0/1 indicating TIME_AUGMENTATION
     * @param startPosition  the starting position (0 .. shingleSize -1) for the
     *                       most interesting aspect
     * @param attribution    attribution of the whole shingle
     * @param result         the descriptor to store this in
     */

    void addCurrentTimeAttribution(int baseDimensions, int adjustTime, int startPosition, DiVector attribution,
            AnomalyDescriptor result) {
        double[] flattenedAttribution = new double[baseDimensions - adjustTime];
        for (int i = 0; i < baseDimensions - adjustTime; i++) {
            flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
        }
        result.setCurrentTimeAttribution(flattenedAttribution);
        if (forestMode == ForestMode.TIME_AUGMENTED) {
            result.setTimeAttribution(attribution.getHighLowSum(startPosition + baseDimensions - 1));
        }
    }

    /**
     * core routine which collates the information about the most recent point
     * 
     * @param point          input (shingled) point, ready for RCF
     * @param inputTimeStamp timestamp of input
     * @param inputPoint     actual input point (need not be shingled, or augmented
     *                       with time)
     * @return description containing scores, grade, confidence, expected values,
     *         attribution etc.
     */
    protected AnomalyDescriptor getAnomalyDescription(double[] point, long inputTimeStamp, double[] inputPoint) {
        AnomalyDescriptor result = new AnomalyDescriptor();
        // DiVector attribution = forest.getAnomalyAttribution(point);
        double score = forest.getAnomalyScore(point);
        result.setRcfScore(score);
        int internalTimeStamp = preprocessor.getInternalTimeStamp();
        result.setTotalUpdates(internalTimeStamp);
        result.setTimestamp(inputTimeStamp);
        result.setForestSize(forest.getNumberOfTrees());
        result.setDataConfidence(computeDataConfidence());
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;
        int adjustTime = (forestMode == ForestMode.TIME_AUGMENTED) ? 1 : 0;

        if (score > 0) {
            double[] currentValues = Arrays.copyOf(inputPoint, inputPoint.length);
            result.setCurrentValues(currentValues);
        }

        result.setThreshold(thresholder.threshold());

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
        }

        // the score is now high enough to be considered an anomaly
        inHighScoreRegion = true;
        result.setInHighScoreRegion(true);

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
        // note gap cannot be less or equal 0

        // the forecast may not be reasonable with less data
        boolean reasonableForecast = (internalTimeStamp > MINIMUM_OBSERVATIONS_FOR_EXPECTED)
                && (shingleSize * baseDimensions >= 4);

        if (reasonableForecast && lastAnomalyPoint != null && lastExpectedPoint != null && gap > 0
                && gap <= shingleSize) {
            double correctedScore = applyBasicCorrector(point, gap, shingleSize, baseDimensions);
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

        DiVector attribution = forest.getAnomalyAttribution(point);

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

        int index = maxContribution(attribution, baseDimensions, -shingleSize) + 1;

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
            lastRelativeIndex = index;
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
                lastRelativeIndex = index = 0; // current point
                lastAnomalyPoint = Arrays.copyOf(point, point.length);
                update(score, newScore, true);
            } else {
                // previousIsPotentialAnomaly is true now, but not calling it anomaly either
                update(score, newScore, true);
                result.setAnomalyGrade(0);
                return result;
            }
        }

        result.setAttribution(attribution);
        result.setRelativeIndex(index);
        result.setExpectedValuesPresent(reasonableForecast);
        if (reasonableForecast) {
            addExpectedAndUpdateState(inputPoint, point, newPoint, result, index);
        } else {
            lastExpectedPoint = null;
        }
        addCurrentTimeAttribution(baseDimensions, adjustTime, startPosition, attribution, result);
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

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
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

    double computeDataConfidence() {
        long total = preprocessor.getValuesSeen();
        double lambda = forest.getTimeDecay();
        double totalExponent = total * lambda;
        if (totalExponent == 0) {
            return 0.0;
        } else if (totalExponent >= 20) {
            return Math.min(1.0, preprocessor.getDataQuality().getMean());
        } else {
            double eTotal = Math.exp(totalExponent);
            double confidence = preprocessor.getDataQuality().getMean()
                    * (eTotal - Math.exp(lambda * Math.min(total, forest.getOutputAfter()))) / (eTotal - 1);
            return Math.max(0, confidence);
        }
    }

    /**
     * @return a new builder.
     */
    public static Builder<?> builder() {
        return new Builder<>();
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        protected int dimensions;
        protected int sampleSize = DEFAULT_SAMPLE_SIZE;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected Optional<Integer> startNormalization = Optional.empty();
        protected Optional<Integer> stopNormalization = Optional.empty();
        protected int numberOfTrees = DEFAULT_NUMBER_OF_TREES;
        protected Optional<Double> timeDecay = Optional.empty();
        protected Optional<Double> horizon = Optional.empty();
        protected Optional<Double> lowerThreshold = Optional.empty();
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
        protected TransformMethod transformMethod = TransformMethod.NONE;
        protected ImputationMethod imputationMethod = PREVIOUS;
        protected ForestMode forestMode = ForestMode.STANDARD;
        protected boolean normalizeTime = false;
        protected boolean normalizeValues = false;
        protected double[] fillValues = null;
        protected double[] weights = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected boolean adjustThreshold = false;

        void validate() {
            if (forestMode == ForestMode.TIME_AUGMENTED) {
                if (internalShinglingEnabled.isPresent()) {
                    checkArgument(shingleSize == 1 || internalShinglingEnabled.get(),
                            " shingle size has to be 1 or " + "internal shingling must turned on");
                    checkArgument(transformMethod == TransformMethod.NONE || internalShinglingEnabled.get(),
                            " internal shingling must turned on for transforms");
                } else {
                    internalShinglingEnabled = Optional.of(true);
                }
                if (useImputedFraction.isPresent()) {
                    throw new IllegalArgumentException(" imputation infeasible");
                }
            } else if (forestMode == ForestMode.STREAMING_IMPUTE) {
                checkArgument(shingleSize > 1, "imputation with shingle size 1 is not meaningful");
                internalShinglingEnabled.ifPresent(x -> checkArgument(x, " non-internal shingling requires "
                        + " full shingles : for these there is nothing to imputer "));
                checkArgument(!normalizeTime, " time values are used in imputation");
            } else {
                if (!internalShinglingEnabled.isPresent()) {
                    internalShinglingEnabled = Optional.of(false);
                }
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
            RandomCutForest.Builder builder = new RandomCutForest.Builder().dimensions(dimensions)
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

        public T startNormalization(int startNormalization) {
            this.startNormalization = Optional.of(startNormalization);
            return (T) this;
        }

        public T stopNormalization(int stopNormalization) {
            this.stopNormalization = Optional.of(stopNormalization);
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

        public T imputationMethod(ImputationMethod imputationMethod) {
            this.imputationMethod = imputationMethod;
            return (T) this;
        }

        public T fillValues(double[] values) {
            // values cannot be a null
            this.fillValues = Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T weights(double[] values) {
            // values cannot be a null
            this.weights = Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T normalizeTime(boolean normalizeTime) {
            this.normalizeTime = normalizeTime;
            return (T) this;
        }

        public T transformMethod(TransformMethod method) {
            this.transformMethod = method;
            return (T) this;
        }

        public T forestMode(ForestMode forestMode) {
            this.forestMode = forestMode;
            return (T) this;
        }

        public T adjustThreshold(boolean adjustThreshold) {
            this.adjustThreshold = adjustThreshold;
            return (T) this;
        }

    }
}
