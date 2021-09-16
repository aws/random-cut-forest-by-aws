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
import static com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest.MINIMUM_OBSERVATIONS_FOR_EXPECTED;

import java.util.Arrays;
import java.util.Optional;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class BasicPreprocessor {

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

    protected TransformMethod transformMethod;

    protected int shingleSize;

    protected int dimension;

    protected int inputLength;

    protected ForestMode mode;

    public BasicPreprocessor(Builder<?> builder) {
        inputLength = builder.inputLength;
        dimension = builder.dimensions;
        shingleSize = builder.shingleSize;
        mode = builder.forestMode;
        lastShingledPoint = new double[dimension];
        this.transformMethod = builder.transformMethod;
        this.startNormalization = builder.startNormalization;
        this.stopNormalization = builder.stopNormalization;
        checkArgument(startNormalization <= stopNormalization, "incorrect normalization paramters");
        this.normalizeTime = builder.normalizeTime;
        previousTimeStamps = new long[shingleSize];
        lastShingledInput = new double[shingleSize * inputLength];
        double discount = builder.timeDecay;
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

        if (normalizeTime || transformMethod != TransformMethod.NONE) {
            initialValues = new double[startNormalization][];
            initialTimeStamps = new long[startNormalization];
        }

        if (mode == ForestMode.STREAMING_IMPUTE) {
            imputationMethod = builder.fillin;
            if (imputationMethod == FIXED_VALUES) {
                // checkArgument(transformMethod == TransformMethod.NONE,
                // " transformations and filling with fixed values in actuals are unusual; not
                // supported at the moment");
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

    double[] getScaledInput(double[] input, long timestamp, RandomCutForest forest, double[] factors,
            double timeFactor) {
        double[] scaledInput = transformValues(input, factors);
        if (mode == ForestMode.TIME_AUGMENTED) {
            scaledInput = augmentTime(scaledInput, timestamp, timeFactor);
        } else if (mode == ForestMode.STREAMING_IMPUTE) {
            scaledInput = applyImpute(scaledInput, timestamp, forest);
        }
        return scaledInput;
    }

    /**
     * stores initial data for normalization
     *
     * @param inputPoint input data
     * @param timestamp  timestamp
     */
    void storeInitial(double[] inputPoint, long timestamp) {
        initialTimeStamps[valuesSeen] = timestamp;
        initialValues[valuesSeen] = Arrays.copyOf(inputPoint, inputPoint.length);
        ++valuesSeen;
    }

    protected boolean requireNormalization() {
        return (normalizeTime || transformMethod == TransformMethod.NORMALIZE
                || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE);
    }

    public double[] preProcess(double[] inputPoint, long timestamp, RandomCutForest forest) {
        if (mode == ForestMode.STREAMING_IMPUTE) {
            checkArgument(valuesSeen == 0 || timestamp > previousTimeStamps[forest.getShingleSize() - 1],
                    "incorrect order of time");
        }

        if (requireNormalization()) {
            if (valuesSeen < startNormalization) {
                storeInitial(inputPoint, timestamp);
                return null;
            } else if (valuesSeen == startNormalization) {
                dischargeInitial(forest);
            }
        }
        return getScaledInput(inputPoint, timestamp, forest, null, 0);
    }

    /**
     * updates the state
     *
     * @param inputPoint input actuals
     * @param timeStamp  current stamp
     */
    public void updateState(double[] inputPoint, long timeStamp) {
        for (int i = 0; i < shingleSize - 1; i++) {
            previousTimeStamps[i] = previousTimeStamps[i + 1];
        }
        previousTimeStamps[shingleSize - 1] = timeStamp;

        ++internalTimeStamp;
        if (deviationList != null) {
            for (int i = 0; i < inputPoint.length; i++) {
                double value = inputPoint[i];
                if (transformMethod == TransformMethod.DIFFERENCE
                        || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
                    value -= lastShingledInput[(shingleSize - 1) * inputLength + i];
                }
                deviationList[i].update(value);
            }
        }
        if (inputPoint.length == lastShingledInput.length) {
            lastShingledInput = Arrays.copyOf(inputPoint, inputPoint.length);
        } else {
            shiftLeft(lastShingledInput, inputPoint.length);
            copyAtEnd(lastShingledInput, inputPoint);
        }

    }

    /**
     * an execute once block which first computes the multipliers for normalization
     * and then processes each of the stored inputs
     */
    void dischargeInitial(RandomCutForest forest) {
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            timeStampDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double[] factors = null;
        if (transformMethod == TransformMethod.NORMALIZE || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
            Deviation[] tempList = new Deviation[inputLength];
            for (int j = 0; j < inputLength; j++) {
                tempList[j] = new Deviation(deviationList[j].getDiscount());
            }
            for (int i = 0; i < initialValues.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    double value;
                    if (transformMethod == TransformMethod.NORMALIZE) {
                        value = initialValues[i][j];
                    } else {
                        value = (i == 0) ? 0 : initialValues[i][j] - initialValues[i - 1][j];
                    }
                    tempList[j].update(value);
                }
            }
            factors = new double[inputLength];
            for (int j = 0; j < inputLength; j++) {
                factors[j] = tempList[j].getDeviation();
            }
        }
        for (int i = 0; i < valuesSeen; i++) {
            double[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], forest, factors,
                    timeStampDeviation.getDeviation());
            updateState(initialValues[i], initialTimeStamps[i]);
            // update forest
            if (updateAllowed()) {
                forest.update(scaledInput);
            }
        }
    }

    public boolean updateAllowed() {
        return (mode != ForestMode.STREAMING_IMPUTE || numberOfImputed < useImputedFraction * shingleSize);
    }

    public void setPreviousTimeStamps(long[] values) {
        previousTimeStamps = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    protected void copyAtEnd(double[] array, double[] small) {
        checkArgument(array.length > small.length, " incorrect operation ");
        System.arraycopy(small, 0, array, array.length - small.length, small.length);
    }

    public double[] getLastShingledInput() {
        return copyIfNotnull(lastShingledInput);
    }

    public void setLastShingledInput(double[] point) {
        lastShingledInput = copyIfNotnull(point);
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    protected void shiftLeft(double[] array, int baseDimension) {
        for (int i = 0; i < array.length - baseDimension; i++) {
            array[i] = array[i + baseDimension];
        }
    }

    public long[] getInitialTimeStamps() {
        return (initialTimeStamps == null) ? null : Arrays.copyOf(initialTimeStamps, initialTimeStamps.length);
    }

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

    public Deviation getTimeStampDeviation() {
        return timeStampDeviation;
    }

    public long[] getPreviousTimeStamps() {
        return (previousTimeStamps == null) ? null : Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
    }

    /**
     * maps a value shifted to the current mean or to a relative space
     *
     * @param value     input value of dimension
     * @param deviation statistic
     * @return the normalized value
     */
    protected double normalize(double value, Deviation deviation, double factor) {
        if (deviation.getCount() < 2) {
            return 0;
        }
        double currentFactor = (factor != 0) ? factor : deviation.getDeviation();
        if (value - deviation.getMean() >= 2 * clipFactor * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return clipFactor;
        }
        if (value - deviation.getMean() < -2 * clipFactor * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return -clipFactor;
        } else {
            // deviation cannot be 0
            return (value - deviation.getMean()) / (2 * (currentFactor + DEFAULT_NORMALIZATION_PRECISION));
        }
    }

    public double inverseTransform(double value, int index, double difference) {
        if (transformMethod == TransformMethod.NONE) {
            return value;
        } else if (transformMethod == TransformMethod.SUBTRACT_MA) {
            return value + deviationList[index].getMean();
        } else if (transformMethod == TransformMethod.NORMALIZE) {

            return deviationList[index].getMean()
                    + 2 * value * (deviationList[index].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
        } else if (transformMethod == TransformMethod.DIFFERENCE) {
            return value + difference;
        }
        checkArgument(transformMethod == TransformMethod.NORMALIZE_DIFFERENCE, "incorrect options");
        return difference + deviationList[index].getMean()
                + +2 * value * (deviationList[index].getDeviation() + DEFAULT_NORMALIZATION_PRECISION);
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
     * determines the next point to fed to the model for a missing value
     *
     * @param fillin            strategy of imputation
     * @param baseDimension     number of values to imput
     * @param lastShingledPoint last full entry seen by model
     * @return an array of baseDimension values corresponding to plausible next
     *         input
     */

    protected double[] impute(RandomCutForest forest, ImputationMethod fillin, int baseDimension,
            double[] lastShingledPoint) {
        double[] result = new double[baseDimension];
        if (fillin == ImputationMethod.ZERO) {
            return result;
        }
        if (fillin == FIXED_VALUES) {
            System.arraycopy(defaultFill, 0, result, 0, baseDimension);
            return result;
        }
        int dimension = forest.getDimensions();
        if (fillin == PREVIOUS) {
            System.arraycopy(lastShingledPoint, dimension - baseDimension, result, 0, baseDimension);
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

    /**
     * applies transformations if desired
     *
     * @param inputPoint input point
     * @return a differenced version of the input
     */
    protected double[] transformValues(double[] inputPoint, double[] factors) {
        if (transformMethod == TransformMethod.NONE) {
            return inputPoint;
        }
        double[] input = new double[inputPoint.length];
        if (transformMethod == TransformMethod.SUBTRACT_MA) {
            for (int i = 0; i < inputPoint.length; i++) {
                input[i] = inputPoint[i] - deviationList[i].getMean();
            }
        }
        if (transformMethod == TransformMethod.NORMALIZE) {
            for (int i = 0; i < input.length; i++) {
                input[i] = normalize(inputPoint[i], deviationList[i], (factors == null) ? 0 : factors[i]);
            }
        }
        if (transformMethod == TransformMethod.DIFFERENCE) {
            for (int i = 0; i < input.length; i++) {
                input[i] = (internalTimeStamp == 0) ? 0
                        : inputPoint[i] - lastShingledInput[(shingleSize - 1) * inputLength + i];
            }
        }
        if (transformMethod == TransformMethod.NORMALIZE_DIFFERENCE) {
            for (int i = 0; i < input.length; i++) {
                double value = (internalTimeStamp == 0) ? 0
                        : inputPoint[i] - lastShingledInput[(shingleSize - 1) * inputLength + i];
                input[i] = normalize(value, deviationList[i], (factors == null) ? 0 : factors[i]);
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
     * @param timefactor a variable that controls initial segment (non-zero) vs
     *                   continual (0)
     * @return a tuple with one exta field
     */
    protected double[] augmentTime(double[] normalized, long timestamp, double timefactor) {
        double[] scaledInput = new double[normalized.length + 1];
        System.arraycopy(normalized, 0, scaledInput, 0, normalized.length);
        if (valuesSeen <= 1) {
            scaledInput[normalized.length] = 0;
        } else {
            double timeshift = timestamp - previousTimeStamps[shingleSize - 1];
            scaledInput[normalized.length] = (normalizeTime) ? normalize(timeshift, timeStampDeviation, timefactor)
                    : timeshift;
        }
        return scaledInput;
    }

    /**
     * performs imputation if desired; based on the timestamp estimates the number
     * of discrete gaps the intended use case is small gaps -- for large gaps, one
     * should use time augmentation
     *
     * @param input     (potentially scaled) input, which is ready for the forest
     * @param timestamp current timestamp
     * @return the most recent shingle (after applying the current input)
     */
    protected double[] applyImpute(double[] input, long timestamp, RandomCutForest forest) {

        int baseDimension = dimension / shingleSize;
        if (valuesSeen > 1) {
            int gap = (int) Math
                    .floor((timestamp - previousTimeStamps[shingleSize - 1]) / timeStampDeviation.getMean());
            if (gap >= 1.5) {
                checkArgument(input.length == baseDimension, "error in length");
                for (int i = 0; i < gap - 1; i++) {
                    double[] newPart = impute(forest, imputationMethod, baseDimension, lastShingledPoint);
                    shiftLeft(lastShingledPoint, baseDimension);
                    copyAtEnd(lastShingledPoint, newPart);
                    ++internalTimeStamp;
                    numberOfImputed = Math.min(numberOfImputed + 1, shingleSize);
                    if (updateAllowed()) {
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

    public void postProcess(double[] inputPoint, long timestamp, RandomCutForest forest, double[] scaledInput) {
        if (transformMethod != TransformMethod.NORMALIZE || valuesSeen >= startNormalization) {
            if (timeStampDeviation != null) {
                timeStampDeviation.update(timestamp - previousTimeStamps[shingleSize - 1]);
            }
            ++valuesSeen;
            updateState(inputPoint, timestamp);
            if (updateAllowed()) {
                forest.update(scaledInput);
            }
        }
    }

    public double[] getDefaultFill() {
        return copyIfNotnull(defaultFill);
    }

    public void setDefaultFill(double[] values) {
        defaultFill = copyIfNotnull(values);
    }

    public double[] getShingledInput(int index) {
        int base = lastShingledInput.length / shingleSize;
        double[] values = new double[base];
        System.arraycopy(lastShingledInput, index * base, values, 0, base);
        return values;
    }

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
        protected ImputationMethod fillin = PREVIOUS;
        protected ForestMode forestMode = ForestMode.STANDARD;
        protected int inputLength;
        protected boolean normalizeTime = false;
        protected double[] fillValues = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected Optional<Deviation[]> deviations = Optional.empty();
        protected Optional<Deviation> timeDeviation = Optional.empty();

        public BasicPreprocessor build() {
            return new BasicPreprocessor(this);
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
            this.fillValues = (values == null) ? null : Arrays.copyOf(values, values.length);
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

        public T setMode(ForestMode forestMode) {
            this.forestMode = forestMode;
            return (T) this;
        }

        public T deviations(Deviation[] deviations) {
            this.deviations = Optional.of(deviations);
            return (T) this;
        }

        public T timeDeviation(Deviation timeDeviation) {
            this.timeDeviation = Optional.of(timeDeviation);
            return (T) this;
        }

    }
}
