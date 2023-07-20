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
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.config.ImputationMethod.FIXED_VALUES;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.parkservices.preprocessor.transform.WeightedTransformer.NUMBER_OF_STATS;
import static java.lang.Math.exp;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.Optional;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.RCFComputeDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.DifferenceTransformer;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.ITransformer;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.NormalizedDifferenceTransformer;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.NormalizedTransformer;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.SubtractMATransformer;
import com.amazon.randomcutforest.parkservices.preprocessor.transform.WeightedTransformer;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class Preprocessor implements IPreprocessor {

    public static double NORMALIZATION_SCALING_FACTOR = 2.0;

    // in case of normalization, uses this constant in denominator to ensure
    // smoothness near 0
    public static double DEFAULT_NORMALIZATION_PRECISION = 1e-3;

    // the number of points to buffer before starting to normalize/gather statistic
    public static int DEFAULT_START_NORMALIZATION = 10;

    // the number at which to stop normalization -- it may not e easy to imagine why
    // this is required
    // but this is comforting to those interested in "stopping" a model from
    // learning continuously
    public static int DEFAULT_STOP_NORMALIZATION = Integer.MAX_VALUE;

    // in case of normalization the deviations beyond 10 Sigma are likely measure 0
    // events
    public static int DEFAULT_CLIP_NORMALIZATION = 100;

    // normalization is not turned on by default
    public static boolean DEFAULT_NORMALIZATION = false;

    // differencing is not turned on by default
    // for some smooth predictable data differencing is helpful, but have unintended
    // consequences
    public static boolean DEFAULT_DIFFERENCING = false;

    // the fraction of data points that can be imputed in a shingle before the
    // shingle is admitted in a forest
    public static double DEFAULT_USE_IMPUTED_FRACTION = 0.5;

    // minimum number of observations before using a model to predict any expected
    // behavior -- if we can score, we should predict
    public static int MINIMUM_OBSERVATIONS_FOR_EXPECTED = 100;

    public static int DEFAULT_DATA_QUALITY_STATES = 1;

    // the input corresponds to timestamp data and this statistic helps align input
    protected Deviation[] timeStampDeviations;

    // normalize time difference;
    protected boolean normalizeTime;

    protected double weightTime;

    protected double transformDecay;

    // recording the last seen timestamp
    protected long[] previousTimeStamps;

    // this parameter is used as a clock if imputing missing values in the input
    // this is different from valuesSeen in STREAMING_IMPUTE
    protected int internalTimeStamp = 0;

    // initial values used for normalization
    protected double[][] initialValues;
    protected long[] initialTimeStamps;

    // initial values after which to start normalization
    protected int startNormalization;

    // sequence number to stop normalization at
    protected Integer stopNormalization;

    // a number indicating the actual values seen (not imputed)
    protected int valuesSeen = 0;

    // to use a set of default values for imputation
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

    // shingle size in the forest
    protected int shingleSize;

    // actual dimension of the forest
    protected int dimension;

    // length of input to be seen, may depend on internal/external shingling
    protected int inputLength;

    // the mode of the forest used in this preprocessing
    protected ForestMode mode;

    // measures the data quality in imputed modes
    protected Deviation[] dataQuality;

    protected ITransformer transformer;

    public Preprocessor(Builder<?> builder) {
        checkArgument(builder.transformMethod != null, "transform required");
        checkArgument(builder.forestMode != null, " forest mode is required");
        checkArgument(builder.inputLength > 0, "incorrect input length");
        checkArgument(builder.shingleSize > 0, "incorrect shingle size");
        checkArgument(builder.dimensions > 0, "incorrect dimensions");
        checkArgument(builder.shingleSize == 1 || builder.dimensions % builder.shingleSize == 0,
                " shingle size should divide the dimensions");
        checkArgument(builder.forestMode == ForestMode.TIME_AUGMENTED || builder.inputLength == builder.dimensions
                || builder.inputLength * builder.shingleSize == builder.dimensions, "incorrect input size");
        checkArgument(
                builder.forestMode != ForestMode.TIME_AUGMENTED
                        || (builder.inputLength + 1) * builder.shingleSize == builder.dimensions,
                "incorrect input size");
        checkArgument(builder.startNormalization <= builder.stopNormalization, "incorrect normalization parameters");
        checkArgument(builder.startNormalization > 0 || !builder.normalizeTime, " start of normalization cannot be 0");
        checkArgument(
                builder.startNormalization > 0 || !(builder.transformMethod == TransformMethod.NORMALIZE
                        || builder.transformMethod == TransformMethod.NORMALIZE_DIFFERENCE),
                " start of normalization cannot be 0 for these transformations");
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
        double[] weights = new double[inputLength];
        Arrays.fill(weights, 1.0);
        if (builder.weights != null) {
            if (builder.weights.length == inputLength) {
                System.arraycopy(builder.weights, 0, weights, 0, inputLength);
                weightTime = builder.weightTime;
            } else {
                System.arraycopy(builder.weights, 0, weights, 0, inputLength);
                weightTime = builder.weights[inputLength];
            }
        } else {
            weightTime = builder.weightTime;
        }
        previousTimeStamps = new long[shingleSize];
        if (inputLength == dimension) {
            lastShingledInput = new double[dimension];
        } else {
            lastShingledInput = new double[shingleSize * inputLength];
        }
        transformDecay = builder.timeDecay;
        dataQuality = builder.dataQuality.orElse(new Deviation[] { new Deviation(transformDecay) });

        Deviation[] deviationList = new Deviation[NUMBER_OF_STATS * inputLength];
        manageDeviations(deviationList, builder.deviations, transformDecay);
        timeStampDeviations = new Deviation[NUMBER_OF_STATS];
        manageDeviations(timeStampDeviations, builder.timeDeviations, transformDecay);

        if (transformMethod == TransformMethod.NONE) {
            for (int i = 0; i < inputLength; i++) {
                checkArgument(weights[i] == 1.0, "incorrect weights");
            }
            transformer = new WeightedTransformer(weights, deviationList);
        } else if (transformMethod == TransformMethod.WEIGHTED) {
            transformer = new WeightedTransformer(weights, deviationList);
        } else if (transformMethod == TransformMethod.DIFFERENCE) {
            transformer = new DifferenceTransformer(weights, deviationList);
        } else if (transformMethod == TransformMethod.SUBTRACT_MA) {
            transformer = new SubtractMATransformer(weights, deviationList);
        } else if (transformMethod == TransformMethod.NORMALIZE) {
            transformer = new NormalizedTransformer(weights, deviationList);
        } else {
            transformer = new NormalizedDifferenceTransformer(weights, deviationList);
        }

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
            }
            this.useImputedFraction = builder.useImputedFraction.orElse(0.5);
        }
    }

    // the following fills the first argument as copies of the original
    // but if the original is null or otherwise then new deviations are created; the
    // last third
    // are filled with 0.1 * transformDecay and are reserved for smoothing
    void manageDeviations(Deviation[] deviationList, Optional<Deviation[]> original, double timeDecay) {
        checkArgument(deviationList.length % NUMBER_OF_STATS == 0, " has to be a multiple of five");
        int usedDeviations = 0;
        if (original.isPresent()) {
            Deviation[] list = original.get();
            usedDeviations = min(list.length, deviationList.length);
            // note the lengths can be different based on a different version of the model
            // we will convert the model; and rely on RCF's ability to adjust to new data
            for (int i = 0; i < usedDeviations; i++) {
                deviationList[i] = list[i].copy();
            }
        }
        for (int i = usedDeviations; i < deviationList.length - 2 * deviationList.length / 5; i++) {
            deviationList[i] = new Deviation(timeDecay);
        }
        usedDeviations = max(usedDeviations, deviationList.length - 2 * deviationList.length / 5);
        for (int i = usedDeviations; i < deviationList.length; i++) {
            deviationList[i] = new Deviation(0.1 * timeDecay);
        }
    }

    /**
     * decides if normalization is required, and then is used to store and discharge
     * an initial segment
     *
     * @return a boolean indicating th need to store initial values
     */
    public static boolean requireInitialSegment(boolean normalizeTime, TransformMethod transformMethod,
            ForestMode mode) {
        return (normalizeTime || transformMethod == TransformMethod.NORMALIZE
                || transformMethod == TransformMethod.NORMALIZE_DIFFERENCE)
                || transformMethod == TransformMethod.SUBTRACT_MA;
    }

    /**
     * sets up the AnomalyDescriptor object
     *
     * @param description           description of the input point
     * @param lastAnomalyDescriptor the descriptor of the last anomaly
     * @param forest                the RCF
     * @return the descriptor to be used for anomaly scoring
     */
    <P extends AnomalyDescriptor> P initialSetup(P description, RCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {
        description.setForestMode(mode);
        description.setTransformMethod(transformMethod);
        description.setImputationMethod(imputationMethod);
        description.setNumberOfTrees(forest.getNumberOfTrees());
        description.setTotalUpdates(forest.getTotalUpdates());
        description.setLastAnomalyInternalTimestamp(lastAnomalyDescriptor.getInternalTimeStamp());
        description.setLastExpectedRCFPoint(lastAnomalyDescriptor.getExpectedRCFPoint());
        description.setDataConfidence(forest.getTimeDecay(), valuesSeen, forest.getOutputAfter(),
                dataQuality[0].getMean());
        description.setShingleSize(shingleSize);
        description.setInputLength(inputLength);
        description.setDimension(dimension);
        // the adjustments ensure that external and internal shingling always follow the
        // same path note that the preprocessor performs the shingling for
        // STREAMING_IMPUTE
        long adjustedInternal = internalTimeStamp + (forest.isInternalShinglingEnabled() ? 0 : shingleSize - 1);
        int dataDimension = forest.isInternalShinglingEnabled() || mode == ForestMode.STREAMING_IMPUTE
                ? inputLength * shingleSize
                : inputLength;
        description
                .setReasonableForecast((adjustedInternal > MINIMUM_OBSERVATIONS_FOR_EXPECTED) && (dataDimension >= 4));
        description.setScale(getScale());
        description.setShift(getShift());
        description.setDeviations(getSmoothedDeviations());
        return description;
    }

    /**
     * a generic preprocessing invoked by ThresholdedRandomCutForest
     *
     * @param description           description of the input point (so far)
     * @param lastAnomalyDescriptor the descriptor of the last anomaly
     * @param forest                RCF
     * @return the initialized AnomalyDescriptor with the actual RCF point filled in
     *         (could be a result of multiple transformations)
     */
    public <P extends AnomalyDescriptor> P preProcess(P description, RCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        initialSetup(description, lastAnomalyDescriptor, forest);

        double[] inputPoint = description.getCurrentInput();
        long timestamp = description.getInputTimestamp();
        double[] scaledInput = getScaledInput(inputPoint, timestamp, null, getTimeShift());

        if (scaledInput == null) {
            return description;
        }

        double[] point;
        if (forest.isInternalShinglingEnabled()) {
            point = toDoubleArray(forest.transformToShingledPoint(toFloatArray(scaledInput)));
        } else {
            int dimension = forest.getDimensions();
            if (scaledInput.length == dimension) {
                point = scaledInput;
            } else {
                point = new double[dimension];
                System.arraycopy(getLastShingledPoint(), scaledInput.length, point, 0, dimension - scaledInput.length);
                System.arraycopy(scaledInput, 0, point, dimension - scaledInput.length, scaledInput.length);
            }
        }
        if (description.getMissingValues() != null) {
            int[] missing = new int[description.getMissingValues().length];
            for (int i = 0; i < missing.length; i++) {
                missing[i] = description.getMissingValues()[i] + dimension - scaledInput.length + i;
            }
            point = forest.imputeMissingValues(point, missing.length, missing);
        }
        description.setRCFPoint(point);
        description.setInternalTimeStamp(internalTimeStamp); // no impute
        description.setNumberOfNewImputes(0);
        return description;
    }

    public double[] getScale() {
        if (mode != ForestMode.TIME_AUGMENTED) {
            return transformer.getScale();
        } else {
            double[] scale = new double[inputLength + 1];
            System.arraycopy(transformer.getScale(), 0, scale, 0, inputLength);
            scale[inputLength] = (weightTime == 0) ? 0 : 1.0 / weightTime;
            if (normalizeTime) {
                scale[inputLength] *= NORMALIZATION_SCALING_FACTOR
                        * (getTimeGapDifference() + DEFAULT_NORMALIZATION_PRECISION);
            }
            return scale;
        }
    }

    public double[] getShift() {
        double[] previous = (inputLength == lastShingledInput.length) ? lastShingledInput
                : getShingledInput(shingleSize - 1);
        if (mode != ForestMode.TIME_AUGMENTED) {
            return transformer.getShift(previous);
        } else {
            double[] shift = new double[inputLength + 1];
            System.arraycopy(transformer.getShift(previous), 0, shift, 0, inputLength);
            // time is always differenced
            shift[inputLength] = ((normalizeTime) ? getTimeShift() : 0) + previousTimeStamps[shingleSize - 1];
            return shift;
        }
    }

    public double[] getSmoothedDeviations() {
        if (mode != ForestMode.TIME_AUGMENTED) {
            double[] deviations = new double[2 * inputLength];
            System.arraycopy(transformer.getSmoothedDeviations(), 0, deviations, 0, inputLength);
            System.arraycopy(transformer.getSmoothedDifferenceDeviations(), 0, deviations, inputLength, inputLength);
            return deviations;
        } else {
            double[] deviations = new double[2 * inputLength + 2];
            System.arraycopy(transformer.getSmoothedDeviations(), 0, deviations, 0, inputLength);
            System.arraycopy(transformer.getSmoothedDifferenceDeviations(), 0, deviations, inputLength + 1,
                    inputLength);
            // time is differenced (for now) or unchanged
            deviations[inputLength + 1] = timeStampDeviations[4].getMean();
            deviations[2 * inputLength + 1] = timeStampDeviations[4].getMean();
            return deviations;
        }
    }

    /**
     * a generic postprocessor which updates all the state
     * 
     * @param result the descriptor of the evaluation on the current point
     * @param forest the resident RCF
     * @return the descriptor (mutated and augmented appropriately)
     */
    public <P extends AnomalyDescriptor> P postProcess(P result, RCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        double[] point = result.getRCFPoint();
        if (point == null) {
            return result;
        }

        if (result.getAnomalyGrade() > 0) {
            populateAnomalyDescriptorDetails(result);
        }

        double[] inputPoint = result.getCurrentInput();
        long timestamp = result.getInputTimestamp();

        updateState(inputPoint, point, timestamp, previousTimeStamps[shingleSize - 1]);
        ++valuesSeen;
        dataQuality[0].update(1.0);
        if (forest.isInternalShinglingEnabled()) {
            int length = inputLength + ((mode == ForestMode.TIME_AUGMENTED) ? 1 : 0);
            double[] scaledInput = new double[length];
            System.arraycopy(point, point.length - length, scaledInput, 0, length);
            forest.update(scaledInput);
        } else {
            forest.update(point);
        }
        if (result.getAnomalyGrade() > 0) {
            double[] postShift = getShift();
            result.setPostShift(postShift);
            result.setTransformDecay(transformDecay);
        }
        // after the insertions
        result.setPostDeviations(getSmoothedDeviations());
        return result;
    }

    /**
     * adds information of expected point to the result descriptor (provided it is
     * marked anomalous) Note that is uses relativeIndex; that is, it can determine
     * that the anomaly occurred in the past (but within the shingle) and not at the
     * current point -- even though the detection has triggered now While this may
     * appear to be improper, information theoretically we may have a situation
     * where an anomaly is only discoverable after the "horse has bolted" -- suppose
     * that we see a random mixture of the triples { 1, 2, 3} and {2, 4, 5}
     * corresponding to "slow weeks" and "busy weeks". For example 1, 2, 3, 1, 2, 3,
     * 2, 4, 5, 1, 2, 3, 2, 4, 5, ... etc. If we see { 2, 2, X } (at positions 0 and
     * 1 (mod 3)) and are yet to see X, then we can infer that the pattern is
     * anomalous -- but we cannot determine which of the 2's are to blame. If it
     * were the first 2, then the detection is late. If X = 3 then we know it is the
     * first 2 in that unfinished triple; and if X = 5 then it is the second 2. In a
     * sense we are only truly wiser once the bolted horse has returned! But if we
     * were to say that the anomaly was always at the second 2 then that appears to
     * be suboptimal -- one natural path can be based on the ratio of the triples {
     * 1, 2, 3} and {2, 4, 5} seen before. Even better, we can attempt to estimate a
     * dynamic time dependent ratio -- and that is what RCF would do.
     *
     * @param result the description of the current point
     */
    protected void populateAnomalyDescriptorDetails(AnomalyDescriptor result) {
        int base = dimension / shingleSize;
        double[] reference = result.getCurrentInput();
        double[] point = result.getRCFPoint();
        double[] newPoint = result.getExpectedRCFPoint();

        int index = result.getRelativeIndex();
        if (index < 0) {
            reference = getShingledInput(shingleSize + index);
            result.setPastTimeStamp(getTimeStamp(shingleSize + index));
        }
        result.setPastValues(reference);
        if (newPoint != null) {
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
     * maps the time back. The returned value is an approximation for
     * relativePosition less than 0 which corresponds to an anomaly in the past.
     * Since the state of the statistic is now changed based on more recent values
     *
     * @param gap              estimated value
     * @param relativePosition how far back in the shingle
     * @return transform of the time value to original input space
     */
    protected long inverseMapTime(double gap, int relativePosition) {
        // note this corresponds to differencing being always on
        checkArgument(shingleSize + relativePosition >= 0, " error");
        return inverseMapTimeValue(gap, previousTimeStamps[shingleSize - 1 + relativePosition]);
    }

    // same as inverseMapTime, using explicit value also useful in forecast
    protected long inverseMapTimeValue(double gap, long timestamp) {
        double factor = (weightTime == 0) ? 0 : 1.0 / weightTime;
        if (factor == 0) {
            return 0;
        }
        if (normalizeTime) {
            return (long) Math
                    .round(timestamp + getTimeShift() + NORMALIZATION_SCALING_FACTOR * gap * getTimeScale() * factor);
        } else {
            return (long) Math.round(gap * factor + timestamp);
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
    protected double[] getExpectedValue(int relativeBlockIndex, double[] reference, double[] point, double[] newPoint) {
        int base = dimension / shingleSize;
        int startPosition = (shingleSize - 1 + relativeBlockIndex) * base;
        if (mode == ForestMode.TIME_AUGMENTED) {
            --base;
        }
        double[] values = invert(base, startPosition, relativeBlockIndex, newPoint);

        for (int i = 0; i < base; i++) {
            double currentValue = (reference.length == base) ? reference[i] : reference[startPosition + i];
            values[i] = (point[startPosition + i] == newPoint[startPosition + i]) ? currentValue : values[i];
        }
        return values;
    }

    /**
     * inverts the values to the input space from the RCF space
     * 
     * @param base               the number of baseDimensions (often inputLength,
     *                           unless time augmented)
     * @param startPosition      the position in the vector to invert
     * @param relativeBlockIndex the relative blockIndex (related to the position)
     * @param newPoint           the vector
     * @return the values [startPosition, startPosition + base -1] which would
     *         (approximately) produce newPoint
     */
    protected double[] invert(int base, int startPosition, int relativeBlockIndex, double[] newPoint) {
        double[] values = new double[base];
        System.arraycopy(newPoint, startPosition, values, 0, base);
        return transformer.invert(values, getShingledInput(shingleSize - 1 + relativeBlockIndex));
    }

    /**
     *
     * The function inverts the forecast generated by a TRCF model; however as is
     * clear from below, the inversion mandates that the different transformations
     * be handled by different and separate classes. At the same time, different
     * parts of TRCF refer to the same information and thus it makes sense to add
     * this testable function, and refactor with the newer tests present.
     * 
     * @param ranges                the forecast with ranges
     * @param lastAnomalyDescriptor description of last anomaly
     * @return a timed range vector that also contains the time information
     *         corresponding to the forecasts; note that the time values would be 0
     *         for STREAMING_IMPUTE mode
     */
    public TimedRangeVector invertForecastRange(RangeVector ranges, RCFComputeDescriptor lastAnomalyDescriptor) {
        int baseDimension = inputLength + (mode == ForestMode.TIME_AUGMENTED ? 1 : 0);
        checkArgument(ranges.values.length % baseDimension == 0, " incorrect length of ranges");
        int horizon = ranges.values.length / baseDimension;

        int gap = (int) (internalTimeStamp - lastAnomalyDescriptor.getInternalTimeStamp());
        double[] correction = lastAnomalyDescriptor.getDeltaShift();
        if (correction != null) {
            double decay = max(lastAnomalyDescriptor.getTransformDecay(), 1.0 / (3 * shingleSize));
            double factor = exp(-gap * decay);
            for (int i = 0; i < correction.length; i++) {
                correction[i] *= factor;
            }
        } else {
            correction = new double[baseDimension];
        }
        long localTimeStamp = previousTimeStamps[shingleSize - 1];

        TimedRangeVector timedRangeVector;
        if (mode != ForestMode.TIME_AUGMENTED) {
            timedRangeVector = new TimedRangeVector(ranges, horizon);
            // Note that STREAMING_IMPUTE we are already using the time values
            // to fill in values -- moreover such missing values can be large in number
            // predicting next timestamps in the future in such a scenario would correspond
            // to a joint prediction and TIME_AUGMENTED mode may be more suitable.
            // therefore for STREAMING_IMPUTE the timestamps values are not predicted
            if (mode != ForestMode.STREAMING_IMPUTE) {
                double timeGap = getTimeDrift();
                double timeBound = 1.3 * getTimeGapDifference();

                for (int i = 0; i < horizon; i++) {
                    timedRangeVector.timeStamps[i] = inverseMapTimeValue(timeGap, localTimeStamp);
                    timedRangeVector.upperTimeStamps[i] = max(timedRangeVector.timeStamps[i],
                            inverseMapTimeValue(timeGap + timeBound, localTimeStamp));
                    timedRangeVector.lowerTimeStamps[i] = min(timedRangeVector.timeStamps[i],
                            inverseMapTimeValue(max(0, timeGap - timeBound), localTimeStamp));
                    localTimeStamp = timedRangeVector.timeStamps[i];
                }
            }
        } else {
            if (gap <= shingleSize && lastAnomalyDescriptor.getExpectedRCFPoint() != null && gap == 1) {
                localTimeStamp = lastAnomalyDescriptor.getExpectedTimeStamp();
            }
            timedRangeVector = new TimedRangeVector(inputLength * horizon, horizon);
            for (int i = 0; i < horizon; i++) {
                for (int j = 0; j < inputLength; j++) {
                    timedRangeVector.rangeVector.values[i * inputLength + j] = ranges.values[i * baseDimension + j];
                    timedRangeVector.rangeVector.upper[i * inputLength + j] = ranges.upper[i * baseDimension + j];
                    timedRangeVector.rangeVector.lower[i * inputLength + j] = ranges.lower[i * baseDimension + j];
                }
                timedRangeVector.timeStamps[i] = inverseMapTimeValue(
                        max(ranges.values[i * baseDimension + inputLength], 0), localTimeStamp);
                timedRangeVector.upperTimeStamps[i] = max(timedRangeVector.timeStamps[i],
                        inverseMapTimeValue(max(ranges.upper[i * baseDimension + inputLength], 0), localTimeStamp));
                timedRangeVector.lowerTimeStamps[i] = min(timedRangeVector.timeStamps[i],
                        inverseMapTimeValue(max(ranges.lower[i * baseDimension + inputLength], 0), localTimeStamp));
                localTimeStamp = timedRangeVector.upperTimeStamps[i];
            }
        }
        // the following is the post-anomaly transformation, can be impacted by
        // anomalies
        transformer.invertForecastRange(timedRangeVector.rangeVector, inputLength, getShingledInput(shingleSize - 1),
                correction);
        return timedRangeVector;
    }

    /**
     * given an input produces a scaled transform to be used in the forest
     *
     * @param input             the actual input seen
     * @param timestamp         timestamp of said input
     * @param defaults          default statistics, potentially used in
     *                          initialization
     * @param defaultTimeFactor default time statistic
     * @return a scaled/transformed input which can be used in the forest
     */
    protected double[] getScaledInput(double[] input, long timestamp, Deviation[] defaults, double defaultTimeFactor) {
        double[] previous = (input.length == lastShingledInput.length) ? lastShingledInput
                : getShingledInput(shingleSize - 1);
        double[] scaledInput = transformer.transformValues(internalTimeStamp, input, previous, defaults, clipFactor);
        if (mode == ForestMode.TIME_AUGMENTED) {
            scaledInput = augmentTime(scaledInput, timestamp, defaultTimeFactor);
        }
        return scaledInput;
    }

    /**
     * updates the various shingles
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

    protected void updateTimeStampDeviations(long timestamp, long previous) {
        if (timeStampDeviations != null) {
            timeStampDeviations[0].update(timestamp);
            timeStampDeviations[1].update(timestamp - previous);
            // smoothing - not used currently
            timeStampDeviations[2].update(timeStampDeviations[0].getDeviation());
            timeStampDeviations[3].update(timeStampDeviations[1].getMean());
            timeStampDeviations[4].update(timeStampDeviations[1].getDeviation());
        }
    }

    double getTimeScale() {
        return 1.0 + getTimeGapDifference();
    }

    double getTimeGapDifference() {
        return Math.abs(timeStampDeviations[4].getMean());
    }

    double getTimeShift() {
        return timeStampDeviations[1].getMean();
    }

    double getTimeDrift() {
        return timeStampDeviations[3].getMean();
    }

    /**
     * updates the state of the preprocessor
     * 
     * @param inputPoint  the actual input
     * @param scaledInput the transformed input
     * @param timestamp   the timestamp of the input
     * @param previous    the previous timestamp
     */
    protected void updateState(double[] inputPoint, double[] scaledInput, long timestamp, long previous) {
        updateTimeStampDeviations(timestamp, previous);
        updateTimestamps(timestamp);
        double[] previousInput = (inputLength == lastShingledInput.length) ? lastShingledInput
                : getShingledInput(shingleSize - 1);
        transformer.updateDeviation(inputPoint, previousInput);
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

    // a utility function
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
     * maps a value shifted to the current mean or to a relative space for time
     *
     * @return the normalized value
     */
    protected double normalize(double value, double factor) {
        double currentFactor = (factor != 0) ? factor : getTimeScale();
        if (value - getTimeShift() >= NORMALIZATION_SCALING_FACTOR * clipFactor
                * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return clipFactor;
        }
        if (value - getTimeShift() <= -NORMALIZATION_SCALING_FACTOR * clipFactor
                * (currentFactor + DEFAULT_NORMALIZATION_PRECISION)) {
            return -clipFactor;
        } else {
            // deviation cannot be 0
            return (value - getTimeShift())
                    / (NORMALIZATION_SCALING_FACTOR * (currentFactor + DEFAULT_NORMALIZATION_PRECISION));
        }
    }

    /**
     * augments (potentially normalized) input with time (which is always
     * differenced)
     *
     * @param normalized (potentially normalized) input point
     * @param timestamp  timestamp of current point
     * @param timeFactor a factor used in normalizing time
     * @return a tuple with one extra field
     */
    protected double[] augmentTime(double[] normalized, long timestamp, double timeFactor) {
        double[] scaledInput = new double[normalized.length + 1];
        System.arraycopy(normalized, 0, scaledInput, 0, normalized.length);
        if (valuesSeen <= 1) {
            scaledInput[normalized.length] = 0;
        } else {
            double timeShift = timestamp - previousTimeStamps[shingleSize - 1];
            scaledInput[normalized.length] = weightTime
                    * ((normalizeTime) ? normalize(timeShift, timeFactor) : timeShift);
        }
        return scaledInput;
    }

    // mapper
    public long[] getInitialTimeStamps() {
        return (initialTimeStamps == null) ? null : Arrays.copyOf(initialTimeStamps, initialTimeStamps.length);
    }

    // mapper
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

    // mapper
    public void setInitialValues(double[][] values) {
        if (values == null) {
            initialValues = null;
        } else {
            initialValues = new double[values.length][];
            for (int i = 0; i < values.length; i++) {
                initialValues[i] = copyIfNotnull(values[i]);
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
    public Deviation[] getTimeStampDeviations() {
        return timeStampDeviations;
    }

    // mapper
    public long[] getPreviousTimeStamps() {
        return (previousTimeStamps == null) ? null : Arrays.copyOf(previousTimeStamps, previousTimeStamps.length);
    }

    public Deviation[] getDeviationList() {
        return transformer.getDeviations();
    }

    public double getTimeDecay() {
        return transformDecay;
    }

    /**
     * used in mapper; augments weightTime to the weights array to produce a single
     * array of length inputLength + 1
     */
    public double[] getWeights() {
        double[] basic = transformer.getWeights();
        double[] answer = new double[inputLength + 1];
        Arrays.fill(answer, 1.0);
        if (basic != null) {
            checkArgument(basic.length == inputLength, " incorrect length returned");
            System.arraycopy(basic, 0, answer, 0, inputLength);
        }
        answer[inputLength] = weightTime;
        return answer;
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
        protected Optional<Deviation[]> timeDeviations = Optional.empty();
        protected Optional<Deviation[]> dataQuality = Optional.empty();

        public Preprocessor build() {
            if (forestMode == ForestMode.STREAMING_IMPUTE) {
                return new ImputePreprocessor(this);
            } else if (requireInitialSegment(normalizeTime, transformMethod, forestMode)) {
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
            this.deviations = Optional.ofNullable(deviations);
            return (T) this;
        }

        // mapper
        public T dataQuality(Deviation[] dataQuality) {
            this.dataQuality = Optional.ofNullable(dataQuality);
            return (T) this;
        }

        // mapper
        public T timeDeviations(Deviation[] timeDeviations) {
            this.timeDeviations = Optional.ofNullable(timeDeviations);
            return (T) this;
        }

    }

}
