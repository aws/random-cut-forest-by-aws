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
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
@Getter
@Setter
public class ThresholdedRandomCutForest {

    public static int MINIMUM_OBSERVATIONS_FOR_EXPECTED = 100;

    public static double DEFAULT_NORMALIZATION_PRECISION = 1e-3;

    public static int DEFAULT_START_NORMALIZATION = 10;

    public static int DEFAULT_STOP_NORMALIZATION = Integer.MAX_VALUE;

    public static int DEFAULT_CLIP_NORMALIZATION = 10;

    public static double DEFAULT_USE_IMPUTED_FRACTION = 0.5;

    public static int DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS = 2;

    public static boolean DEFAULT_NORMALIZATION = false;

    public static boolean DEFAULT_DIFFERENCING = false;

    public static double DEFAULT_REPEAT_ANOMALY_Z_FACTOR = 3.0;

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

    // this parameter is used in imputing missing values in the input
    protected int valuesSeen = 0;

    // this parameter is used as a clock if imputing missing values in the input
    // this is different from valuesSeen in STREAMING_IMPUTE
    protected int internalTimeStamp = 0;

    // the input corresponds to timestamp data and this statistic helps align input
    protected Deviation timeStampDeviation;

    // can be used to normalize data
    protected Deviation[] deviationList;

    // for possible normalization
    protected boolean normalizeValues = DEFAULT_NORMALIZATION;

    // for possible martingale transforms
    protected boolean differencing = DEFAULT_DIFFERENCING;

    // recording the last seen timestamp
    protected long[] previousTimeStamps;

    // normalize time difference;
    protected boolean normalizeTime = DEFAULT_NORMALIZATION;

    // fraction of data that should be actual input before they are added to RCF
    protected double useImputedFraction = DEFAULT_USE_IMPUTED_FRACTION;

    // number of imputed values in stored shingle
    protected int numberOfImputed;

    // particular strategy for impute
    protected ImputationMethod imputationMethod = PREVIOUS;

    // forestMode of operation
    protected ForestMode forestMode = ForestMode.STANDARD;

    // for FILL_VALUES
    protected double[] defaultFill;

    // last point
    protected double[] lastShingledPoint;

    // last shingled values (without normalization/change or augmentation by time)
    protected double[] lastShingledInput;

    // initial values used for normalization
    protected double[][] initialValues;
    protected long[] initialTimeStamps;

    // initial values after which to start normalization
    protected int startNormalization = DEFAULT_START_NORMALIZATION;

    // sequence number to stop normalization at
    protected int stopNormalization = DEFAULT_STOP_NORMALIZATION;

    // used in normalization
    protected double clipFactor = DEFAULT_CLIP_NORMALIZATION;

    // used for confidence
    protected int lastReset;

    protected RandomCutForest forest;
    protected IThresholder thresholder;

    public ThresholdedRandomCutForest(Builder<?> builder) {
        int inputLength;
        forestMode = builder.forestMode;
        previousTimeStamps = new long[builder.shingleSize];
        lastShingledInput = new double[builder.dimensions];
        this.differencing = builder.differencing;
        this.normalizeValues = builder.normalizeValues;

        if (builder.forestMode == ForestMode.TIME_AUGMENTED) {
            inputLength = builder.dimensions / builder.shingleSize;
            builder.dimensions += builder.shingleSize;
            normalizeTime = builder.normalizeTime;
            // force internal shingling for this option
            builder.internalShinglingEnabled = Optional.of(true);
        } else if (builder.forestMode == ForestMode.STREAMING_IMPUTE) {
            inputLength = builder.dimensions / builder.shingleSize;
            this.imputationMethod = builder.fillin;
            numberOfImputed = builder.shingleSize;
            builder.internalShinglingEnabled = Optional.of(false);
            if (this.imputationMethod == FIXED_VALUES) {
                checkArgument(!normalizeValues,
                        "normalization and filling with fixed values in actuals are unusual; not supported");
                int baseDimension = builder.dimensions / builder.shingleSize;
                // shingling will be performed in this layer and not in forest
                // so that we control admittance of imputed shingles
                checkArgument(builder.fillValues != null && builder.fillValues.length == baseDimension,
                        " the number of values should match the shingled input");
                this.defaultFill = Arrays.copyOf(builder.fillValues, builder.fillValues.length);
            } else if (imputationMethod == ZERO) {
                checkArgument(!normalizeValues,
                        "normalization and filling with zero values in actuals are unusual; not supported");
            }
            this.useImputedFraction = builder.useImputedFraction.orElse(0.5);
        } else {
            boolean smallInput = builder.internalShinglingEnabled.orElse(DEFAULT_INTERNAL_SHINGLING_ENABLED);
            inputLength = (smallInput) ? builder.dimensions / builder.shingleSize : builder.dimensions;
        }

        lastShingledPoint = new double[builder.dimensions];
        double discount = builder.timeDecay
                .orElse(1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY * builder.sampleSize));

        if (this.normalizeValues) {
            deviationList = new Deviation[inputLength];
            for (int i = 0; i < inputLength; i++) {
                deviationList[i] = new Deviation(discount);
            }
        }
        timeStampDeviation = new Deviation(discount);
        forest = builder.buildForest();

        if (normalizeTime || normalizeValues) {
            int number = forest.getOutputAfter() + forest.getShingleSize() - 1 + ((differencing) ? 1 : 0);
            startNormalization = builder.startNormalization.orElse(number);
            checkArgument(startNormalization >= forest.getShingleSize() + ((differencing) ? 1 : 0),
                    " use startNormalization() with n at least" + (forest.getShingleSize() + ((differencing) ? 1 : 0)));
            stopNormalization = builder.stopNormalization.orElse(DEFAULT_STOP_NORMALIZATION);
            checkArgument(startNormalization <= stopNormalization, "normalization stops before start");
            initialValues = new double[startNormalization][];
            initialTimeStamps = new long[startNormalization];
        }

        BasicThresholder basic = new BasicThresholder(builder.anomalyRate);
        if (forest.getDimensions() / forest.getShingleSize() == 1) {
            basic.setLowerThreshold(BasicThresholder.DEFAULT_LOWER_THRESHOLD_ONED);
            basic.setHorizon(BasicThresholder.DEFAULT_HORIZON_ONED);
        }
        thresholder = basic;
    }

    public ThresholdedRandomCutForest(RandomCutForest forest, IThresholder thresholder, Deviation deviation,
            Deviation[] deviations, long[] initialTimeStamps, double[][] initialValues) {
        this.forest = forest;
        this.thresholder = thresholder;
        this.timeStampDeviation = deviation;
        this.deviationList = deviations;
        if (initialTimeStamps != null) {
            checkArgument(initialValues == null || initialTimeStamps.length == initialValues.length, "incorrect input");
            startNormalization = initialTimeStamps.length;
            this.initialTimeStamps = Arrays.copyOf(initialTimeStamps, startNormalization);
        }
        if (initialValues != null) {
            this.initialValues = new double[initialValues.length][];
            for (int i = 0; i < startNormalization; i++) {
                this.initialValues[i] = Arrays.copyOf(initialValues[i], initialValues[i].length);
            }
        }
    }

    /**
     * applies differencing to input (if desired) uses the state of last shingled
     * input
     * 
     * @param inputPoint input point
     * @return a differenced version of the input
     */
    protected double[] applyDifferencing(double[] inputPoint) {
        double[] input = new double[inputPoint.length];
        for (int i = 0; i < inputPoint.length; i++) {
            input[i] = (valuesSeen <= 1) ? 0
                    : inputPoint[i] - lastShingledInput[lastShingledInput.length - inputPoint.length + i];
        }
        return input;
    }

    /**
     * applies normalization to input points
     * 
     * @param input the input tuple (potentially differenced)
     * @return either a normalized tuple or a copy of input
     */
    protected double[] applyNormalization(double[] input) {
        checkArgument(input.length == deviationList.length, " mismatch in length");
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = normalize(input[i], deviationList[i]);
        }
        return normalized;
    }

    /**
     * augments (potentially normalized) input with time (which is always
     * differenced)
     * 
     * @param normalized (potentially normalized) input point
     * @param timestamp  timestamp of current point
     * @return a tuple with one exta field
     */
    public double[] augmentTime(double[] normalized, long timestamp) {
        double[] scaledInput = new double[normalized.length + 1];
        System.arraycopy(normalized, 0, scaledInput, 0, normalized.length);
        if (valuesSeen <= 1) {
            scaledInput[normalized.length] = 0;
        } else {
            double timeshift = timestamp - previousTimeStamps[forest.getShingleSize() - 1];
            scaledInput[normalized.length] = (normalizeTime) ? normalize(timeshift, timeStampDeviation) : timeshift;
        }
        return scaledInput;
    }

    /**
     * performs imputation if deried; based on the timestamp estimates the number of
     * discrete gaps the intended use case is small gaps -- for large gaps, one
     * should use time augmentation
     * 
     * @param input     (potentially scaled) input, which is ready for the forest
     * @param timestamp current timestamp
     * @return the most recent shingle (after applying the current input)
     */
    protected double[] applyImpute(double[] input, long timestamp) {
        int shingleSize = forest.getShingleSize();
        int dimension = forest.getDimensions();
        int baseDimension = dimension / shingleSize;
        if (valuesSeen > 1) {
            int gap = (int) Math
                    .floor((timestamp - previousTimeStamps[shingleSize - 1]) / timeStampDeviation.getMean());
            if (gap >= 1.5) {
                checkArgument(input.length == baseDimension, "error in length");
                for (int i = 0; i < gap - 1; i++) {
                    double[] newPart = impute(imputationMethod, baseDimension, lastShingledPoint);
                    shiftLeft(lastShingledPoint, baseDimension);
                    copyAtEnd(lastShingledPoint, newPart);
                    ++internalTimeStamp;
                    numberOfImputed = Math.min(numberOfImputed + 1, forest.getShingleSize());
                    if (numberOfImputed < useImputedFraction * forest.getShingleSize()) {
                        forest.update(lastShingledPoint);
                    }
                }
            }
        }

        shiftLeft(lastShingledPoint, baseDimension);
        copyAtEnd(lastShingledPoint, input);
        numberOfImputed = Math.max(0, numberOfImputed - 1);
        return Arrays.copyOf(lastShingledPoint, dimension);
    }

    /**
     * updates statistics based on (potentially differenced) input
     * 
     * @param input     (potentially differenced) input
     * @param timeStamp current timestamp
     */
    protected void updateDeviation(double[] input, long timeStamp) {
        if (valuesSeen > 0) {
            timeStampDeviation.update(timeStamp - previousTimeStamps[forest.getShingleSize() - 1]);
        }
        if (!differencing || valuesSeen > 0) {
            if (normalizeValues && valuesSeen < stopNormalization) {
                for (int i = 0; i < input.length; i++) {
                    deviationList[i].update(normalize(input[i], deviationList[i]));
                }
            }
        }
    }

    /**
     * updates the state
     * 
     * @param inputPoint input actuals
     * @param timeStamp  current stamp
     */
    protected void updateState(double[] inputPoint, long timeStamp) {
        int shingleSize = forest.getShingleSize();

        for (int i = 0; i < shingleSize - 1; i++) {
            previousTimeStamps[i] = previousTimeStamps[i + 1];
        }
        previousTimeStamps[shingleSize - 1] = timeStamp;

        if (inputPoint.length == lastShingledInput.length) {
            lastShingledInput = Arrays.copyOf(inputPoint, inputPoint.length);
        } else {
            shiftLeft(lastShingledInput, inputPoint.length);
            copyAtEnd(lastShingledInput, inputPoint);
        }
    }

    /**
     * the function is used with normalization (in time or in values), we store the
     * initial few values where RCF would return a 0 score anyways. We then use the
     * statistics of this overall group to normalize data. A fully streaming
     * normalization would have large variations in the initial segment.
     */
    void dischargeInitial() {
        if (initialTimeStamps != null) {
            for (int i = 0; i < initialTimeStamps.length - 1; i++) {
                timeStampDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
            }

        }
        if (initialValues != null && normalizeValues) {
            if (differencing) {
                for (int i = 0; i < initialValues.length - 1; i++) {
                    for (int j = 0; j < initialValues[i].length; j++) {
                        deviationList[j].update(initialValues[i + 1][j] - initialValues[i][j]);
                    }
                }
            } else {
                for (int i = 0; i < initialValues.length; i++) {
                    for (int j = 0; j < initialValues[i].length; j++) {
                        deviationList[j].update(initialValues[i][j]);
                    }
                }
            }
        }
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
        if (forestMode == ForestMode.STREAMING_IMPUTE) {
            checkArgument(valuesSeen == 0 || timestamp > previousTimeStamps[forest.getShingleSize() - 1],
                    "incorrect order of time");
        }
        if (normalizeTime || normalizeValues) {
            if (valuesSeen < startNormalization) {
                initialTimeStamps[valuesSeen] = timestamp;
                initialValues[valuesSeen] = Arrays.copyOf(inputPoint, inputPoint.length);
                ++valuesSeen;
                return new AnomalyDescriptor();
            } else {
                if (valuesSeen == startNormalization) {
                    dischargeInitial();
                    for (int i = 0; i < valuesSeen; i++) {
                        continuousProcess(initialValues[i], initialTimeStamps[i], false);
                    }
                }
            }
        }
        return continuousProcess(inputPoint, timestamp, true);
    }

    protected AnomalyDescriptor continuousProcess(double[] inputPoint, long timestamp, boolean getResult) {
        double[] input = (differencing) ? applyDifferencing(inputPoint) : inputPoint;

        double[] scaledInput = (normalizeValues) ? applyNormalization(input) : input;

        if (forestMode == ForestMode.TIME_AUGMENTED) {
            scaledInput = augmentTime(scaledInput, timestamp);
        } else if (forestMode == ForestMode.STREAMING_IMPUTE) {
            scaledInput = applyImpute(scaledInput, timestamp);
        }

        AnomalyDescriptor result = null;
        if (getResult) {
            // the following handles both external and internal shingling
            double[] point = forest.transformToShingledPoint(scaledInput);

            result = getAnomalyDescription(point, timestamp, inputPoint);

            // update state
            updateDeviation(input, timestamp);
            ++valuesSeen;
            ++internalTimeStamp;
        }

        updateState(inputPoint, timestamp);
        // update forest
        if (forestMode != ForestMode.STREAMING_IMPUTE
                || numberOfImputed < useImputedFraction * forest.getShingleSize()) {
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
        double[] reference = inputPoint;
        double[] newPoint = expected;

        if (index < 0 && result.isStartOfAnomaly()) {
            // anomaly in the past and detected late; repositioning the computation
            // index 0 is current time
            startPosition = shingleSize * baseDimensions + (result.getRelativeIndex() - 1) * baseDimensions;

            int[] likelyMissingIndices = largestFeatures(result.getAttribution(), startPosition, baseDimensions,
                    numberOfAttributors);
            newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
            int base = baseDimensions - adjustTime;
            double[] oldValues = new double[base];
            System.arraycopy(lastShingledInput, (shingleSize - 1 + result.getRelativeIndex()) * base, oldValues, 0,
                    base);
            reference = oldValues;
            result.setOldValues(oldValues);
            result.setRelativeIndex(index);
            result.setOldTimeStamp(previousTimeStamps[shingleSize - 1 + result.getRelativeIndex()]);
        } else {
            result.setRelativeIndex(0);
        }

        if (forestMode == ForestMode.TIME_AUGMENTED) {
            result.setExpectedTimeStamp(inverseMapTime(newPoint[startPosition + baseDimensions - 1],
                    result.getRelativeIndex(), normalizeTime));
        }

        double[] values = new double[baseDimensions - adjustTime];
        for (int i = 0; i < baseDimensions - adjustTime; i++) {
            values[i] = (point[startPosition + i] == newPoint[startPosition + i]) ? reference[i]
                    : denormalize(newPoint[startPosition + i], normalizeValues, differencing, deviationList, i);
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
            if (differencing) {
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
        DiVector attribution = forest.getAnomalyAttribution(point);
        double score = attribution.getHighLowSum();
        result.setRcfScore(score);
        result.setTotalUpdates(internalTimeStamp);
        result.setTimestamp(inputTimeStamp);
        result.setForestSize(forest.getNumberOfTrees());
        result.setAttribution(attribution);
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        int startPosition = (shingleSize - 1) * baseDimensions;
        int adjustTime = (forestMode == ForestMode.TIME_AUGMENTED) ? 1 : 0;

        if (score > 0) {
            double[] currentValues = Arrays.copyOf(inputPoint, inputPoint.length);
            result.setCurrentValues(currentValues);
        }

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

    public Deviation getTimeStampDeviation() {
        return timeStampDeviation;
    }

    public void setLastExpectedPoint(double[] point) {
        lastExpectedPoint = copyIfNotnull(point);
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    /**
     * maps a value shifted to the current mean or to a relative space
     * 
     * @param value     input value of dimension
     * @param deviation statistic
     * @return the normalized value
     */
    protected double normalize(double value, Deviation deviation) {
        if (deviation.getCount() < 2) {
            return 0;
        }
        if (value - deviation.getMean() >= 2 * clipFactor
                * (deviation.getDeviation() + DEFAULT_NORMALIZATION_PRECISION)) {
            return clipFactor;
        }
        if (value - deviation.getMean() < -2 * clipFactor
                * (deviation.getDeviation() + DEFAULT_NORMALIZATION_PRECISION)) {
            return -clipFactor;
        } else {
            // deviation cannot be 0
            return (value - deviation.getMean()) / (2 * (deviation.getDeviation() + DEFAULT_NORMALIZATION_PRECISION));
        }
    }

    /**
     * inverse of above map
     * 
     * @param gap        observed value
     * @param normalize  (flag indicating normalization
     * @param difference are values
     * @param deviations statistic of attribute
     * @param index      attribute number
     * @return potential value
     */
    protected double denormalize(double gap, boolean normalize, boolean difference, Deviation[] deviations, int index) {
        int base = lastShingledInput.length / forest.getShingleSize();
        int position = (forest.getShingleSize() - 1) * base;
        if (normalize) {
            double value = deviations[index].getMean()
                    + 2 * gap * (deviations[index].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
            return (difference) ? value + lastShingledInput[position + index] : value;
        } else {
            return (difference) ? gap + lastShingledInput[position + index] : gap;
        }
    }

    /**
     * maps the time back. The returned value is an approximation for
     * relativePosition less than 0 which corresponds to an anomaly in the past.
     * Since the state of the statistic is now changed based on more recent values
     * 
     * @param gap              estimated value
     * @param relativePosition how far back in the shingle
     * @param normalize        the flag that determines normalization
     * @return transform of the time value to original input space
     */
    protected long inverseMapTime(double gap, int relativePosition, boolean normalize) {
        // note this ocrresponds to differencing being always on
        checkArgument(forest.getShingleSize() + relativePosition >= 0, " error");
        int shingleSize = forest.getShingleSize();
        if (normalize) {
            return (long) Math.floor(previousTimeStamps[shingleSize - 1 + relativePosition]
                    + timeStampDeviation.getMean() + 2 * gap * timeStampDeviation.getDeviation());
        } else {
            return (long) Math
                    .floor(gap + previousTimeStamps[shingleSize - 1 + relativePosition] + timeStampDeviation.getMean());
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

    protected double[] impute(ImputationMethod fillin, int baseDimension, double[] lastShingledPoint) {
        double[] result = new double[baseDimension];
        if (fillin == ImputationMethod.ZERO) {
            if (differencing) {
                System.arraycopy(lastShingledPoint, lastShingledPoint.length - baseDimension, result, 0, baseDimension);
                for (int i = 0; i < baseDimension; i++) {
                    result[i] = -result[i]; // subtraction from 0
                }
            }
            return result;
        }
        if (fillin == FIXED_VALUES) {
            System.arraycopy(defaultFill, 0, result, 0, baseDimension);
            if (differencing) {
                for (int i = 0; i < baseDimension; i++) {
                    result[i] -= lastShingledPoint[i]; // subtraction from 0
                }
            }
            return result;
        }
        int dimension = forest.getDimensions();
        if (fillin == PREVIOUS) {
            if (!differencing) {
                System.arraycopy(lastShingledPoint, dimension - baseDimension, result, 0, baseDimension);
            }
            return result;
        }
        if (forest.getTotalUpdates() < MINIMUM_OBSERVATIONS_FOR_EXPECTED || dimension < 4 || baseDimension >= 3) {
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

    protected void shiftLeft(double[] array, int baseDimension) {
        for (int i = 0; i < array.length - baseDimension; i++) {
            array[i] = array[i + baseDimension];
        }
    }

    protected long[] getInitialTimeStamps() {
        return (initialTimeStamps == null) ? null : Arrays.copyOf(initialTimeStamps, initialTimeStamps.length);
    }

    protected double[][] getInitialValues() {
        if (initialValues == null) {
            return null;
        } else {
            double[][] result = new double[initialValues.length][];
            for (int i = 0; i < initialValues.length; i++) {
                result[i] = copyIfNotnull(initialValues[i]);
            }
            return result;
        }

    }

    protected long[] getPreviousTimeStamps() {
        return (previousTimeStamps == null) ? null : Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
    }

    protected void setPreviousTimeStamps(long[] values) {
        previousTimeStamps = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    protected void copyAtEnd(double[] array, double[] small) {
        checkArgument(array.length > small.length, " incorrect operation ");
        System.arraycopy(small, 0, array, array.length - small.length, small.length);
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

    public void setZfactor(double factor) {
        BasicThresholder t = (BasicThresholder) thresholder;
        t.setZfactor(factor);
        triggerFactor = Math.max(factor, triggerFactor);
    }

    public void setLowerThreshold(double lower) {
        BasicThresholder t = (BasicThresholder) thresholder;
        t.setLowerThreshold(lower);
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
        protected ImputationMethod fillin = PREVIOUS;
        protected ForestMode forestMode = ForestMode.STANDARD;
        protected boolean normalizeTime = false;
        protected boolean normalizeValues = false;
        protected double[] fillValues = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected boolean differencing = false;

        void validate() {
            if (forestMode == ForestMode.TIME_AUGMENTED) {
                if (internalShinglingEnabled.isPresent()) {
                    checkArgument(shingleSize == 1 || internalShinglingEnabled.get(),
                            " shingle size has to be 1 or " + "internal shingling must turned on");
                    checkArgument(!differencing || internalShinglingEnabled.get(),
                            " internal shingling must turned on for differencing");
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

        public T fillIn(ImputationMethod imputationMethod) {
            this.fillin = imputationMethod;
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

        public T differencing(boolean differencing) {
            this.differencing = differencing;
            return (T) this;
        }

        public T normalizeValues(boolean normalizeValues) {
            this.normalizeValues = normalizeValues;
            return (T) this;
        }

        public T setMode(ForestMode forestMode) {
            this.forestMode = forestMode;
            return (T) this;
        }

    }
}
