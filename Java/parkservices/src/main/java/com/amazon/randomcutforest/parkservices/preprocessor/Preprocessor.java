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

package com.amazon.randomcutforest.parkservices.preprocessor;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.config.ImputationMethod.ZERO;
import static com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest.DEFAULT_USE_IMPUTED_FRACTION;

import java.util.Arrays;
import java.util.Optional;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.DiVector;

@Getter
@Setter
public class Preprocessor {

    public static double DEFAULT_NORMALIZATION_PRECISION = 1e-3;

    public static int DEFAULT_START_NORMALIZATION = 10;

    public static int DEFAULT_STOP_NORMALIZATION = Integer.MAX_VALUE;

    public static int DEFAULT_CLIP_NORMALIZATION = 10;

    public static boolean DEFAULT_NORMALIZATION = false;

    public static boolean DEFAULT_DIFFERENCING = false;

    // can be used to normalize data
    protected Deviation[] deviationList;

    // the input corresponds to timestamp data and this statistic helps align input
    protected Deviation timeStampDeviation;

    // normalize time difference;
    protected boolean normalizeTime;

    protected boolean augmentTime;

    // recording the last seen timestamp
    protected long[] previousTimeStamps;

    // this parameter is used as a clock if imputing missing values in the input
    // this is different from valuesSeen in STREAMING_IMPUTE
    protected int internalTimeStamp = 0;

    // initial values used for normalization
    protected double[][] initialValues;
    protected long[] initialTimeStamps;

    // initial values after which to start normalization
    protected int startNormalization = DEFAULT_START_NORMALIZATION;

    // sequence number to stop normalization at
    protected Integer stopNormalization = DEFAULT_STOP_NORMALIZATION;

    protected int valuesSeen = 0;

    // for FILL_VALUES
    protected double[] defaultFill;

    // fraction of data that should be actual input before they are added to RCF
    protected double useImputedFraction = DEFAULT_USE_IMPUTED_FRACTION;

    // number of imputed values in stored shingle
    protected int numberOfImputed;

    // particular strategy for impute
    protected ImputationMethod imputationMethod = PREVIOUS;

    // used in normalization
    protected double clipFactor = DEFAULT_CLIP_NORMALIZATION;

    // last shingled values (without normalization/change or augmentation by time)
    protected double[] lastShingledInput;

    // last point
    protected double[] lastShingledPoint;

    // method used to transform data in the preprocessor
    protected TransformMethod transformMethod;

    // shiongle size in the forest
    protected int shingleSize;

    // actual dimension of the forest
    protected int dimension;

    // length of input to be seen, may depend on internal/external shingling
    protected int inputLength;

    // weights to be used for WEIGHTED transformation
    protected double[] weights;

    // the mode of the forest used in this preprocessing
    protected ForestMode mode;

    // measures the data quality in imputed modes
    protected Deviation dataQuality;

    protected long lastActualInternal;

    protected long lastInputTimeStamp;

    public Preprocessor(Builder<?> builder) {
        checkArgument(builder.transformMethod != null, "transform required");
        checkArgument(builder.forestMode != null, " forest mode is required");
        checkArgument(builder.inputLength > 0, "incorrect input length");
        checkArgument(builder.shingleSize > 0, "incorrect shingle size");
        checkArgument(builder.dimensions > 0, "incorrect dimensions");
        checkArgument(builder.shingleSize == 1 || builder.dimensions % builder.shingleSize == 0,
                " shingle size should divide the dimensions");
        checkArgument(builder.forestMode == ForestMode.TIME_AUGMENTED || builder.inputLength == builder.dimensions
                || builder.inputLength * builder.shingleSize == builder.dimensions, "incorrect inputsize");
        checkArgument(
                builder.forestMode != ForestMode.TIME_AUGMENTED
                        || (builder.inputLength + 1) * builder.shingleSize == builder.dimensions,
                "incorrect inputsize");
        checkArgument(builder.startNormalization <= builder.stopNormalization, "incorrect normalization paramters");
        checkArgument(builder.startNormalization > 0 || !builder.normalizeTime, " start of normalization cannot be 0");
        checkArgument(
                builder.startNormalization > 0 || !(builder.transformMethod == TransformMethod.NORMALIZE
                        || builder.transformMethod == TransformMethod.NORMALIZE_DIFFERENCE),
                " start of normalization cannot be 0 for these transformations");
        checkArgument(
                builder.transformMethod != TransformMethod.WEIGHTED
                        || builder.weights != null && builder.weights.length >= builder.inputLength,
                " incorrect weights");
        checkArgument(builder.weights == null || builder.weights.length >= builder.inputLength, " incorrect weights");
        inputLength = builder.inputLength;
        dimension = builder.dimensions;
        shingleSize = builder.shingleSize;
        mode = builder.forestMode;
        lastShingledPoint = new double[dimension];
        this.transformMethod = builder.transformMethod;
        this.startNormalization = builder.startNormalization;
        this.stopNormalization = builder.stopNormalization;
        this.normalizeTime = builder.normalizeTime;
        this.weights = new double[inputLength + 1];
        Arrays.fill(weights, 1);
        if (builder.weights != null) {
            if (builder.weights.length == inputLength) {
                System.arraycopy(builder.weights, 0, weights, 0, inputLength);
                weights[inputLength] = builder.weightTime;
            } else {
                System.arraycopy(builder.weights, 0, weights, 0, inputLength + 1);
            }
        } else {
            weights[inputLength] = builder.weightTime;
        }
        previousTimeStamps = new long[shingleSize];
        if (inputLength == dimension) {
            lastShingledInput = new double[dimension];
        } else {
            lastShingledInput = new double[shingleSize * inputLength];
        }
        double discount = builder.timeDecay;
        dataQuality = builder.dataQuality.orElse(new Deviation(discount));

        if (this.transformMethod != TransformMethod.NONE && this.transformMethod != TransformMethod.DIFFERENCE) {
            if (builder.deviations.isPresent()) {
                deviationList = builder.deviations.get();
            } else {
                deviationList = new Deviation[inputLength];
                for (int i = 0; i < inputLength; i++) {
                    deviationList[i] = new Deviation(discount);
                }
            }
        }
        timeStampDeviation = builder.timeDeviation.orElse(new Deviation(discount));

        if (mode == ForestMode.STREAMING_IMPUTE) {
            imputationMethod = builder.imputationMethod;
            normalizeTime = true;
            if (imputationMethod == FIXED_VALUES) {
                int baseDimension = builder.dimensions / builder.shingleSize;
                // shingling will be performed in this layer and not in forest
                // so that we control admittance of imputed shingles
                checkArgument(builder.fillValues != null && builder.fillValues.length == baseDimension,
                        " the number of values should match the shingled input");
                this.defaultFill = Arrays.copyOf(builder.fillValues, builder.fillValues.length);
            } else if (imputationMethod == ZERO) {
                // checkArgument(builder.transformMethod == TransformMethod.NONE,
                // "transformations and filling with zero values in actuals are unusual; not
                // supported at the moment");
            }
            this.useImputedFraction = builder.useImputedFraction.orElse(0.5);
        }
    }

    /**
     * A generic preprocessing call
     * 
     * @param inputPoint           the actual input
     * @param timestamp            the timestamp of the corresponding input
     * @param forest               the RCF in use
     * @param lastAnomalyTimeStamp the timestamp of the last anomaly, useful in
     *                             imputation and in future can be used in
     *                             transformations
     * @param lastExpectedValue    the expected value (in the space of the RCF
     *                             shingle)
     * @return a shingled/unshingled transformed point (based on configurations) to
     *         be used in scoring
     */
    public double[] preProcess(double[] inputPoint, long timestamp, RandomCutForest forest, long lastAnomalyTimeStamp,
            double[] lastExpectedValue) {
        lastActualInternal = internalTimeStamp;
        lastInputTimeStamp = previousTimeStamps[shingleSize - 1];
        return getScaledInput(inputPoint, timestamp);
    }

    /**
     * adds information of expected point to the result descriptor (provided it is
     * marked anomalous) Note that is uses relativeIndex; that is, it can determine
     * that the anomaly occurred in the past (but within the shingle) and not at the
     * current point -- even though the detection has triggered now While this may
     * appear to be improper, information theoretically we may have a situation
     * where an anomaly is only discoverable after the "horse has bolted" -- suppose
     * that we see a random mixture of the triples { 1, 2, 3} and {2, 4, 5}
     * correpsonding to "slow weeks" and "busy weeks". For example 1, 2, 3, 1, 2, 3,
     * 2, 4, 5, 1, 2, 3, 2, 4, 5, ... etc. If we see { 2, 2, X } (at positions 0 and
     * 1 (mod 3)) and are yet to see X, then we can infer that the pattern is
     * anomalous -- but we cannot determine which of the 2's are to blame. If it
     * were the first 2, then the detection is late. If X = 3 then we know it is the
     * first 2 in that unfinished triple; and if X = 5 then it is the second 2. In a
     * sense we are only truly wiser once the bolted horse has returned! But if we
     * were to say that the anomaly was always at the seocnd 2 then that appears to
     * be suboptimal -- one natural path can be based on the ratio of the triples {
     * 1, 2, 3} and {2, 4, 5} seen before. Even better, we can attempt to estimate a
     * dynamic time dependent ratio -- and that is what RCF would do.
     *
     * @param result the description of the current point
     */
    protected void addRelevantAttribution(AnomalyDescriptor result) {
        int base = dimension / shingleSize;
        int startPosition = (shingleSize - 1 + result.getRelativeIndex()) * base;
        DiVector attribution = result.getAttribution();
        if (mode == ForestMode.TIME_AUGMENTED) {
            --base;
        }
        double[] flattenedAttribution = new double[base];

        for (int i = 0; i < base; i++) {
            flattenedAttribution[i] = attribution.getHighLowSum(startPosition + i);
        }
        result.setRelevantAttribution(flattenedAttribution);
        if (mode == ForestMode.TIME_AUGMENTED) {
            result.setTimeAttribution(attribution.getHighLowSum(startPosition + base));
        }
    }

    /**
     * a generic postprocessor which updates all the state
     * 
     * @param result     the descriptor of the evaluation on the current point
     * @param inputPoint the current input point
     * @param timestamp  the timestamp of the current input
     * @param forest     the resident RCF
     * @return the descriptor (mutated and augmented appropriately)
     */
    public AnomalyDescriptor postProcess(AnomalyDescriptor result, double[] inputPoint, long timestamp,
            RandomCutForest forest) {

        double[] point = result.getRcfPoint();
        checkArgument(point != null, " should not be postprocessing");
        if (result.getAnomalyGrade() > 0) {
            double[] reference = inputPoint;
            double[] newPoint = result.getExpectedRCFPoint();

            int index = result.getRelativeIndex();

            if (newPoint != null) {
                if (index < 0 && result.isStartOfAnomaly()) {
                    reference = getShingledInput(shingleSize + index);
                    result.setOldValues(reference);
                    result.setOldTimeStamp(getTimeStamp(shingleSize - 1 + index));
                }
                if (mode == ForestMode.TIME_AUGMENTED) {
                    int endPosition = (shingleSize - 1 + index + 1) * dimension / shingleSize;
                    double timeGap = (newPoint[endPosition - 1] - point[endPosition - 1]);
                    long expectedTimestamp = (timeGap == 0) ? getTimeStamp(shingleSize - 1 + index)
                            : inverseMapTime(timeGap, index);
                    result.setExpectedTimeStamp(expectedTimestamp);
                }
                double[] values = getExpectedValue(index, reference, point, newPoint);
                result.setExpectedValues(0, values, 1.0);
            }

            addRelevantAttribution(result);
        }

        dataQuality.update(1.0);
        updateState(inputPoint, point, timestamp);
        if (timeStampDeviation != null) {
            timeStampDeviation.update(timestamp - previousTimeStamps[shingleSize - 1]);
        }
        ++valuesSeen;
        if (forest.isInternalShinglingEnabled()) {
            int length = inputLength + ((mode == ForestMode.TIME_AUGMENTED) ? 1 : 0);
            double[] scaledInput = new double[length];
            System.arraycopy(point, point.length - length, scaledInput, 0, length);
            forest.update(scaledInput);
        } else {
            forest.update(point);
        }
        return result;
    }

    /**
     * maps the time back. The returned value is an approximation for
     * relativePosition less than 0 which corresponds to an anomaly in the past.
     * Since the state of the statistic is now changed based on more recent values
     *
     * @param gap              estimated value
     * @param relativePosition how far back in the shingle
     * @return transform of the time value to original input space
     */
    public long inverseMapTime(double gap, int relativePosition) {
        // note this ocrresponds to differencing being always on
        checkArgument(shingleSize + relativePosition >= 0, " error");
        if (normalizeTime) {
            return (long) Math.floor(previousTimeStamps[shingleSize - 1 + relativePosition]
                    + timeStampDeviation.getMean() + 2 * gap * timeStampDeviation.getDeviation());
        } else {
            return (long) Math
                    .floor(gap + previousTimeStamps[shingleSize - 1 + relativePosition] + timeStampDeviation.getMean());
        }
    }

    /**
     * returns the input values corresponding to a position in the shingle; this is
     * needed in the corrector steps; and avoids the need for replicating this
     * information downstream
     * 
     * @param index position in the shingle
     * @return the input values for those positions in the shingle
     */
    public double[] getShingledInput(int index) {
        int base = lastShingledInput.length / shingleSize;
        double[] values = new double[base];
        System.arraycopy(lastShingledInput, index * base, values, 0, base);
        return values;
    }

    /**
     * produces the expected value given location of the anomaly -- being aware that
     * the nearest anomaly may be behind us in time.
     * 
     * @param relativeBlockIndex the relative index of the anomaly
     * @param reference          the reference input (so that we do not generate
     *                           arbitrary rounding errors of transformations which
     *                           can be indistinguishable from true expected values)
     * @param point              the point (in the RCF shingled space)
     * @param newPoint           the expected point (in the RCF shingled space) --
     *                           where only the most egregiously offending entries
     *                           corresponding to the shingleSize - 1 +
     *                           relativeBlockIndex are changed.
     * @return the set of values (in the input space) that would have produced
     *         newPoint
     */
    public double[] getExpectedValue(int relativeBlockIndex, double[] reference, double[] point, double[] newPoint) {
        int base = dimension / shingleSize;
        int startPosition = (shingleSize - 1 + relativeBlockIndex) * base;
        if (mode == ForestMode.TIME_AUGMENTED) {
            --base;
        }
        double[] values = new double[base];

        for (int i = 0; i < base; i++) {
            double currentValue = (reference.length == base) ? reference[i] : reference[startPosition + i];
            values[i] = (point[startPosition + i] == newPoint[startPosition + i]) ? currentValue
                    : inverseTransform(newPoint[startPosition + i], i, relativeBlockIndex);
        }
        return values;
    }

    /**
     * if we find an estimated value for input index i, then this function inverts
     * that estimate to indicate (approximately) what that value should have been in
     * the actual input space
     *
     * @param value              estimated value
     * @param index              position in the input vector
     * @param relativeBlockIndex the index of the block in the shingle
     * @return the estimated value whose transform would be the value
     */
    protected double inverseTransform(double value, int index, int relativeBlockIndex) {
        if (transformMethod == TransformMethod.NONE) {
            return value;
        } else if (transformMethod == TransformMethod.WEIGHTED) {
            return (weights[index] == 0) ? 0 : value / weights[index];
        } else if (transformMethod == TransformMethod.SUBTRACT_MA) {
            return (weights[index] == 0) ? 0 : (value + deviationList[index].getMean()) / weights[index];
        }
        double[] difference = getShingledInput(shingleSize - 1 + relativeBlockIndex);
        checkArgument(transformMethod == TransformMethod.DIFFERENCE, "incorrect configuration");
        return (weights[index] == 0) ? 0 : (value + difference[index]) / weights[index];
    }

    /**
     * given an input produces a scaled transform to be used in the forest
     * 
     * @param input     the actual input seen
     * @param timestamp timestamp of said input
     * @return a scaled/transformed input which can be used in the forest
     */
    protected double[] getScaledInput(double[] input, long timestamp) {
        double[] scaledInput = transformValues(input);
        if (mode == ForestMode.TIME_AUGMENTED) {
            scaledInput = augmentTime(scaledInput, timestamp);
        }
        return scaledInput;
    }

    /**
     * decides if normalization is required, and then is used to store and discharge
     * an initial segment
     * 
     * @return a boolean indicating th need to store initial values
     */
    public static boolean requireInitialSegment(boolean normalizeTime, TransformMethod transformMethod) {
        return (normalizeTime || transformMethod == TransformMethod.NORMALIZE
                || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE);
    }

    /**
     * updates the varios shingles
     * 
     * @param inputPoint  the input point
     * @param scaledPoint the scaled/transformed point which is used in the RCF
     */

    protected void updateShingle(double[] inputPoint, double[] scaledPoint) {
        if (inputPoint.length == lastShingledInput.length) {
            lastShingledInput = Arrays.copyOf(inputPoint, inputPoint.length);
        } else {
            shiftLeft(lastShingledInput, inputPoint.length);
            copyAtEnd(lastShingledInput, inputPoint);
        }
        if (scaledPoint.length == lastShingledPoint.length) {
            lastShingledPoint = Arrays.copyOf(scaledPoint, scaledPoint.length);
        } else {
            shiftLeft(lastShingledPoint, scaledPoint.length);
            copyAtEnd(lastShingledPoint, scaledPoint);
        }
    }

    /**
     * updates timestamps
     * 
     * @param timestamp the timestamp of the current input
     */
    protected void updateTimestamps(long timestamp) {
        for (int i = 0; i < shingleSize - 1; i++) {
            previousTimeStamps[i] = previousTimeStamps[i + 1];
        }
        previousTimeStamps[shingleSize - 1] = timestamp;
        ++internalTimeStamp;
    }

    /**
     * updates deviations which are used in some of the transformations (and would
     * be null for others)
     * 
     * @param inputPoint the input point
     */
    void updateDeviation(double[] inputPoint) {
        for (int i = 0; i < inputPoint.length; i++) {
            double value = inputPoint[i];
            if (transformMethod == TransformMethod.DIFFERENCE) {
                value -= lastShingledInput[lastShingledInput.length - inputLength + i];
            }
            deviationList[i].update(value);
        }
    }

    /**
     * updates the state, correspoding to timestamps, the deviations, and the
     * shingles
     *
     * @param inputPoint input actuals
     * @param timestamp  current stamp
     */
    protected void updateState(double[] inputPoint, double[] scaledInput, long timestamp) {
        updateTimestamps(timestamp);
        if (deviationList != null) {
            updateDeviation(inputPoint);
        }
        updateShingle(inputPoint, scaledInput);
    }

    /**
     * copies at the end for a shingle
     * 
     * @param array shingled array
     * @param small new small array
     */
    protected void copyAtEnd(double[] array, double[] small) {
        checkArgument(array.length > small.length, " incorrect operation ");
        System.arraycopy(small, 0, array, array.length - small.length, small.length);
    }

    // an utility function
    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    // left shifting used for the shingles
    protected void shiftLeft(double[] array, int baseDimension) {
        for (int i = 0; i < array.length - baseDimension; i++) {
            array[i] = array[i + baseDimension];
        }
    }

    /**
     * applies transformations if desired
     *
     * @param inputPoint input point
     * @return a differenced version of the input
     */
    protected double[] transformValues(double[] inputPoint) {
        if (transformMethod == TransformMethod.NONE) {
            return inputPoint;
        }
        double[] input = new double[inputPoint.length];
        if (transformMethod == TransformMethod.WEIGHTED) {
            for (int i = 0; i < inputPoint.length; i++) {
                input[i] = inputPoint[i] * weights[i];
            }
        } else if (transformMethod == TransformMethod.SUBTRACT_MA) {
            for (int i = 0; i < inputPoint.length; i++) {
                input[i] = (internalTimeStamp == 0) ? 0 : weights[i] * (inputPoint[i] - deviationList[i].getMean());
            }
        } else if (transformMethod == TransformMethod.DIFFERENCE) {
            for (int i = 0; i < input.length; i++) {
                input[i] = (internalTimeStamp == 0) ? 0
                        : weights[i] * (inputPoint[i] - lastShingledInput[lastShingledInput.length - inputLength + i]);
            }
        }
        return input;
    }

    /**
     * augments (potentially normalized) input with time (which is always
     * differenced)
     *
     * @param normalized (potentially normalized) input point
     * @param timestamp  timestamp of current point
     * @return a tuple with one exta field
     */
    protected double[] augmentTime(double[] normalized, long timestamp) {
        double[] scaledInput = new double[normalized.length + 1];
        System.arraycopy(normalized, 0, scaledInput, 0, normalized.length);
        if (valuesSeen <= 1) {
            scaledInput[normalized.length] = 0;
        } else {
            double timeshift = timestamp - previousTimeStamps[shingleSize - 1];
            scaledInput[normalized.length] = weights[inputLength] * timeshift;
        }
        return scaledInput;
    }

    // mapper
    public long[] getInitialTimeStamps() {
        return (initialTimeStamps == null) ? null : Arrays.copyOf(initialTimeStamps, initialTimeStamps.length);
    }

    public void setInitialTimeStamps(long[] values) {
        initialTimeStamps = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    // mapper
    public double[][] getInitialValues() {
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

    public void setInitialValues(double[][] values) {
        if (values == null) {
            initialValues = null;
        } else {
            initialValues = new double[values.length][];
            for (int i = 0; i < values.length; i++) {
                initialValues[i] = copyIfNotnull(initialValues[i]);
            }
        }
    }

    // mapper
    public double[] getLastShingledInput() {
        return copyIfNotnull(lastShingledInput);
    }

    // mapper
    public void setLastShingledInput(double[] point) {
        lastShingledInput = copyIfNotnull(point);
    }

    // mapper
    public void setPreviousTimeStamps(long[] values) {
        if (values == null) {
            numberOfImputed = shingleSize;
            previousTimeStamps = null;
        } else {
            checkArgument(values.length == shingleSize, " incorrect length ");
            previousTimeStamps = Arrays.copyOf(values, values.length);
            numberOfImputed = 0;
            for (int i = 0; i < previousTimeStamps.length - 1; i++) {
                if (previousTimeStamps[i] == previousTimeStamps[i + 1]) {
                    ++numberOfImputed;
                }
            }
        }
    }

    // mapper
    public Deviation getTimeStampDeviation() {
        return timeStampDeviation;
    }

    // mapper
    public long[] getPreviousTimeStamps() {
        return (previousTimeStamps == null) ? null : Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
    }

    // mapper
    public double[] getWeights() {
        return copyIfNotnull(weights);
    }

    // mapper/semisupervision
    public void setWeights(double[] values) {
        weights = copyIfNotnull(values);
    }

    // mapper
    public double[] getDefaultFill() {
        return copyIfNotnull(defaultFill);
    }

    // mapper
    public void setDefaultFill(double[] values) {
        defaultFill = copyIfNotnull(values);
    }

    // mapper
    public long getTimeStamp(int index) {
        return previousTimeStamps[index];
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
        protected int startNormalization = DEFAULT_START_NORMALIZATION;
        protected Integer stopNormalization = DEFAULT_STOP_NORMALIZATION;
        protected double timeDecay;
        protected Optional<Long> randomSeed = Optional.empty();
        protected int shingleSize = DEFAULT_SHINGLE_SIZE;
        protected double anomalyRate = 0.01;
        protected TransformMethod transformMethod = TransformMethod.NONE;
        protected ImputationMethod imputationMethod = PREVIOUS;
        protected ForestMode forestMode = ForestMode.STANDARD;
        protected int inputLength;
        protected boolean normalizeTime = false;
        protected double[] fillValues = null;
        protected double[] weights = null;
        protected double weightTime = 1.0;
        protected ThresholdedRandomCutForest thresholdedRandomCutForest = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected Optional<Deviation[]> deviations = Optional.empty();
        protected Optional<Deviation> timeDeviation = Optional.empty();
        protected Optional<Deviation> dataQuality = Optional.empty();

        public Preprocessor build() {
            if (forestMode == ForestMode.STREAMING_IMPUTE) {
                return new ImputePreprocessor(this);
            } else if (requireInitialSegment(normalizeTime, transformMethod)) {
                return new InitialSegmentPreprocessor(this);
            }
            return new Preprocessor(this);
        }

        public T dimensions(int dimensions) {
            this.dimensions = dimensions;
            return (T) this;
        }

        public T inputLength(int inputLength) {
            this.inputLength = inputLength;
            return (T) this;
        }

        public T startNormalization(int startNormalization) {
            this.startNormalization = startNormalization;
            return (T) this;
        }

        public T stopNormalization(Integer stopNormalization) {
            this.stopNormalization = stopNormalization;
            return (T) this;
        }

        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        public T timeDecay(double timeDecay) {
            this.timeDecay = timeDecay;
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

        public T imputationMethod(ImputationMethod imputationMethod) {
            this.imputationMethod = imputationMethod;
            return (T) this;
        }

        public T fillValues(double[] values) {
            // values can be null
            this.fillValues = (values == null) ? null : Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T weights(double[] values) {
            // values can be null
            this.weights = (values == null) ? null : Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T weightTime(double value) {
            this.weightTime = value;
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

        // mapper
        public T deviations(Deviation[] deviations) {
            this.deviations = Optional.of(deviations);
            return (T) this;
        }

        // mapper
        public T dataQuality(Deviation dataQuality) {
            this.dataQuality = Optional.of(dataQuality);
            return (T) this;
        }

        // mapper
        public T timeDeviation(Deviation timeDeviation) {
            this.timeDeviation = Optional.of(timeDeviation);
            return (T) this;
        }

        public T thresholder(ThresholdedRandomCutForest thresholdedRandomCutForest) {
            this.thresholdedRandomCutForest = thresholdedRandomCutForest;
            return (T) this;
        }

    }
}
