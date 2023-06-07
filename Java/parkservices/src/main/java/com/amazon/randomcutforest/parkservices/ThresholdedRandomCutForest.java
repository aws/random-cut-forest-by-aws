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
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INITIAL_ACCEPT_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INTERNAL_SHINGLING_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_NUMBER_OF_TREES;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PARALLEL_EXECUTION_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PRECISION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_AUTO_THRESHOLD;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD_NORMALIZED;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_SCORE_DIFFERENCING;
import static java.lang.Math.max;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.IPreprocessor;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.returntypes.RangeVector;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
@Getter
@Setter
public class ThresholdedRandomCutForest {

    // saved description of the last seen anomaly
    IRCFComputeDescriptor lastAnomalyDescriptor;

    // forestMode of operation
    protected ForestMode forestMode = ForestMode.STANDARD;

    protected TransformMethod transformMethod = TransformMethod.NONE;

    protected ScoringStrategy scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;

    protected RandomCutForest forest;

    protected PredictorCorrector predictorCorrector;

    protected IPreprocessor preprocessor;

    public ThresholdedRandomCutForest(Builder<?> builder) {

        forestMode = builder.forestMode;
        transformMethod = builder.transformMethod;
        scoringStrategy = builder.scoringStrategy;
        Preprocessor.Builder<?> preprocessorBuilder = Preprocessor.builder().shingleSize(builder.shingleSize)
                .transformMethod(builder.transformMethod).forestMode(builder.forestMode);

        int inputLength;
        if (builder.forestMode == ForestMode.TIME_AUGMENTED) {
            inputLength = builder.dimensions / builder.shingleSize;
            preprocessorBuilder.inputLength(inputLength);
            builder.dimensions += builder.shingleSize;
            preprocessorBuilder.normalizeTime(builder.normalizeTime);
            // force internal shingling for this option
            builder.internalShinglingEnabled = Optional.of(true);
        } else if (builder.forestMode == ForestMode.STREAMING_IMPUTE) {
            checkArgument(builder.shingleSize > 1, " shingle size 1 is not useful in impute");
            inputLength = builder.dimensions / builder.shingleSize;
            preprocessorBuilder.inputLength(inputLength);

            preprocessorBuilder.imputationMethod(builder.imputationMethod);
            preprocessorBuilder.normalizeTime(true);
            if (builder.fillValues != null) {
                preprocessorBuilder.fillValues(builder.fillValues);
            }
            // forcing external for the forest to control admittance
            builder.internalShinglingEnabled = Optional.of(false);
            preprocessorBuilder.useImputedFraction(builder.useImputedFraction.orElse(0.5));
        } else {
            // STANDARD
            boolean smallInput = builder.internalShinglingEnabled.orElse(DEFAULT_INTERNAL_SHINGLING_ENABLED);
            inputLength = (smallInput) ? builder.dimensions / builder.shingleSize : builder.dimensions;
            preprocessorBuilder.inputLength(inputLength);
        }

        forest = builder.buildForest();
        preprocessorBuilder.weights(builder.weights);
        preprocessorBuilder.weightTime(builder.weightTime.orElse(1.0));
        preprocessorBuilder.timeDecay(builder.transformDecay.orElse(1.0 / builder.sampleSize));

        preprocessorBuilder.dimensions(builder.dimensions);
        preprocessorBuilder
                .stopNormalization(builder.stopNormalization.orElse(Preprocessor.DEFAULT_STOP_NORMALIZATION));
        preprocessorBuilder
                .startNormalization(builder.startNormalization.orElse(Preprocessor.DEFAULT_START_NORMALIZATION));

        preprocessor = preprocessorBuilder.build();
        predictorCorrector = new PredictorCorrector(forest.getTimeDecay(), builder.anomalyRate, builder.adjustThreshold,
                builder.learnNearIgnoreExpected, builder.dimensions / builder.shingleSize,
                builder.randomSeed.orElse(0L));
        lastAnomalyDescriptor = new RCFComputeDescriptor(null, 0, builder.forestMode, builder.transformMethod,
                builder.imputationMethod);

        // multiple (not extremely well correlated) dimensions typically reduce scores
        // normalization reduces scores

        if (builder.dimensions == builder.shingleSize
                || (forestMode == ForestMode.TIME_AUGMENTED) && (builder.dimensions == 2 * builder.shingleSize)) {
            predictorCorrector.setLowerThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD));
        } else {
            if (builder.transformMethod == TransformMethod.NONE
                    || builder.transformMethod == TransformMethod.SUBTRACT_MA) {
                predictorCorrector.setLowerThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD));
            } else {
                predictorCorrector.setLowerThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD_NORMALIZED));
            }
            predictorCorrector.setZfactor(2.5);
        }

        predictorCorrector.setScoreDifferencing(builder.scoreDifferencing.orElse(DEFAULT_SCORE_DIFFERENCING));
        int base = builder.dimensions / builder.shingleSize;
        double[] nearExpected = new double[4 * base];
        builder.ignoreNearExpectedFromAbove.ifPresent(array -> {
            validateNonNegativeArray(array, base);
            System.arraycopy(array, 0, nearExpected, 0, base);
        });
        builder.ignoreNearExpectedFromBelow.ifPresent(array -> {
            validateNonNegativeArray(array, base);
            System.arraycopy(array, 0, nearExpected, base, base);
        });
        builder.ignoreNearExpectedFromAboveByRatio.ifPresent(array -> {
            validateNonNegativeArray(array, base);
            System.arraycopy(array, 0, nearExpected, 2 * base, base);
        });
        builder.ignoreNearExpectedFromBelowByRatio.ifPresent(array -> {
            validateNonNegativeArray(array, base);
            System.arraycopy(array, 0, nearExpected, 3 * base, base);
        });
        predictorCorrector.setIgnoreNearExpected(nearExpected);
    }

    void validateNonNegativeArray(double[] array, int num) {
        checkArgument(array.length == num, "incorrect length");
        for (double element : array) {
            checkArgument(element >= 0, " has to be non-negative");
        }
    }

    // for mappers
    public ThresholdedRandomCutForest(ForestMode forestMode, TransformMethod transformMethod,
            ScoringStrategy scoringStrategy, RandomCutForest forest, PredictorCorrector predictorCorrector,
            Preprocessor preprocessor, RCFComputeDescriptor descriptor) {
        this.forestMode = forestMode;
        this.transformMethod = transformMethod;
        this.scoringStrategy = scoringStrategy;
        this.forest = forest;
        this.predictorCorrector = predictorCorrector;
        this.preprocessor = preprocessor;
        this.lastAnomalyDescriptor = descriptor;
    }

    // for conversion from other thresholding models
    public ThresholdedRandomCutForest(RandomCutForest forest, double futureAnomalyRate, List<Double> values) {
        this.forest = forest;
        int dimensions = forest.getDimensions();
        int inputLength = (forest.isInternalShinglingEnabled()) ? dimensions / forest.getShingleSize()
                : forest.getDimensions();
        Preprocessor preprocessor = new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE)
                .dimensions(dimensions).shingleSize(forest.getShingleSize()).inputLength(inputLength).build();
        this.predictorCorrector = new PredictorCorrector(new BasicThresholder(values, futureAnomalyRate), inputLength);
        preprocessor.setValuesSeen((int) forest.getTotalUpdates());
        preprocessor.getDataQuality()[0].update(1.0);
        this.preprocessor = preprocessor;
        this.lastAnomalyDescriptor = new RCFComputeDescriptor(null, forest.getTotalUpdates());
    }

    /**
     * an extensible function call that applies a preprocess, a core function and
     * the postprocessing corresponding to the preprocess step. It manages the
     * caching strategy of the forest since there are multiple calls to the forest
     * 
     * @param input        an abstract input (which may be mutated)
     * @param preprocessor the preprocessor applied to the input
     * @param core         the core function applied after preprocessing
     * @param <T>          the type of the input
     * @return the final result (switching caching off if needed)
     */
    public <T extends AnomalyDescriptor> T singleStepProcess(T input, IPreprocessor preprocessor, Function<T, T> core) {
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        T answer;
        try {
            if (cacheDisabled) { // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            answer = preprocessor.postProcess(core.apply(preprocessor.preProcess(input, lastAnomalyDescriptor, forest)),
                    lastAnomalyDescriptor, forest);
        } finally {
            if (cacheDisabled) { // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }
        return answer;
    }

    /**
     * a single call that prepreprocesses data, compute score/grade and updates
     * state
     * 
     * @param inputPoint current input point
     * @param timestamp  time stamp of input
     * @return anomaly descriptor for the current input point
     */
    public AnomalyDescriptor process(double[] inputPoint, long timestamp) {
        return process(inputPoint, timestamp, null);
    }

    /**
     * a single call that prepreprocesses data, compute score/grade and updates
     * state when the current input has potentially missing values
     *
     * @param inputPoint    current input point
     * @param timestamp     time stamp of input
     * @param missingValues indices of the input which are missing/questionable
     *                      values
     * @return anomaly descriptor for the current input point
     */
    public AnomalyDescriptor process(double[] inputPoint, long timestamp, int[] missingValues) {

        Function<AnomalyDescriptor, AnomalyDescriptor> function = (x) -> predictorCorrector.detect(x,
                lastAnomalyDescriptor, forest);

        AnomalyDescriptor initial = new AnomalyDescriptor(inputPoint, timestamp);
        initial.setScoringStrategy(scoringStrategy);

        if (missingValues != null) {
            checkArgument(missingValues.length <= inputPoint.length, " incorrect data");
            for (int i = 0; i < missingValues.length; i++) {
                checkArgument(missingValues[i] >= 0 && missingValues[i] < inputPoint.length, " incorrect positions ");
            }
            initial.setMissingValues(missingValues);
        }
        AnomalyDescriptor description = singleStepProcess(initial, preprocessor, function);

        if (description.getAnomalyGrade() > 0) {
            lastAnomalyDescriptor = description.copyOf();
        }
        return description;
    }

    /**
     * a function that extrapolates the data seen by the ThresholdedRCF model, and
     * uses the transformations allowed (as opposed to just using RCFs). The
     * forecasting also allows for predictor-corrector pattern which implies that
     * some noise can be eliminated -- this can be important for various
     * transformations. While the algorithm can function for STREAMING_IMPUTE mode
     * where missing data is imputed on the fly, it may require effort to validate
     * that the internal imputation is reasonably consistent with extrapolation. In
     * general, since the STREAMING_IMPUTE can use non-RCF options to fill in
     * missing data, the internal imputation and extrapolation need not be
     * consistent.
     * 
     * @param horizon    the length of time in the future which is being forecast
     * @param correct    a boolean indicating if predictor-corrector subroutine
     *                   should be turned on; this is specially helpful if there has
     *                   been an anomaly in the recent past
     * @param centrality in general RCF predicts the p50 value of conditional
     *                   samples (centrality = 1). This parameter relaxes the
     *                   conditional sampling. Using assumptions about input data
     *                   (hence external to this code) it may be possible to use
     *                   this parameter and the range information for confidence
     *                   bounds.
     * @return a timed range vector where the values[i] correspond to the forecast
     *         for horizon (i+1). The upper and lower arrays indicate the
     *         corresponding bounds based on the conditional sampling (and
     *         transformation). Note that TRCF manages time in process() and thus
     *         the forecasts always have timestamps associated which makes it easier
     *         to execute the same code for various forest modes such as
     *         STREAMING_IMPUTE, STANDARD and TIME_AUGMENTED. For STREAMING_IMPUTE
     *         the time components of the prediction will be 0 because the time
     *         information is already being used to fill in missing entries. For
     *         STANDARD mode the time components would correspond to average arrival
     *         difference. For TIME_AUGMENTED mode the time componentes would be the
     *         result of the joint prediction. Finally note that setting weight of
     *         time or any of the input columns will also 0 out the corresponding
     *         forecast.
     */

    public TimedRangeVector extrapolate(int horizon, boolean correct, double centrality) {
        // checkArgument(forestMode != ForestMode.STREAMING_IMPUTE, "not yet
        // supported");
        checkArgument(
                (transformMethod != TransformMethod.DIFFERENCE
                        && transformMethod != TransformMethod.NORMALIZE_DIFFERENCE)
                        || horizon <= preprocessor.getShingleSize() / 2 + 1,
                "reduce horizon or use a different transformation, single step differencing will be noisy");
        int shingleSize = preprocessor.getShingleSize();
        checkArgument(shingleSize > 1, "extrapolation is not meaningful for shingle size = 1");
        // note the forest may have external shingling ...
        int dimensions = forest.getDimensions();
        int blockSize = dimensions / shingleSize;
        double[] lastPoint = preprocessor.getLastShingledPoint();
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        RangeVector answer = new RangeVector(horizon * blockSize);
        int gap = (int) (preprocessor.getInternalTimeStamp() - lastAnomalyDescriptor.getInternalTimeStamp());
        try {
            if (cacheDisabled) { // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            float[] newPoint = toFloatArray(lastPoint);

            // gap will be at least 1
            if (gap <= shingleSize && correct && lastAnomalyDescriptor.getExpectedRCFPoint() != null) {
                if (gap == 1) {
                    newPoint = toFloatArray(lastAnomalyDescriptor.getExpectedRCFPoint());
                } else {
                    newPoint = predictorCorrector.applyBasicCorrector(newPoint, gap, shingleSize, blockSize,
                            lastAnomalyDescriptor);
                }
            }
            answer = forest.extrapolateFromShingle(newPoint, horizon, blockSize, centrality);
        } finally {
            if (cacheDisabled) { // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }
        return preprocessor.invertForecastRange(answer, lastAnomalyDescriptor);
    }

    public TimedRangeVector extrapolate(int horizon) {
        return extrapolate(horizon, true, 1.0);
    }

    public RandomCutForest getForest() {
        return forest;
    }

    public void setZfactor(double factor) {
        predictorCorrector.setZfactor(factor);
    }

    public void setLowerThreshold(double lower) {
        predictorCorrector.setLowerThreshold(lower);
    }

    @Deprecated
    public void setHorizon(double horizon) {
        predictorCorrector.setScoreDifferencing(horizon);
    }

    public void setScoreDifferencing(double persistence) {
        predictorCorrector.setScoreDifferencing(persistence);
    }

    @Deprecated
    public void setInitialThreshold(double initial) {
        predictorCorrector.setInitialThreshold(initial);
    }

    public void setIgnoreNearExpected(double[] shift) {
        predictorCorrector.setIgnoreNearExpected(shift);
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
        protected Optional<Double> scoreDifferencing = Optional.empty();
        protected Optional<Double> lowerThreshold = Optional.empty();
        protected Optional<Double> weightTime = Optional.empty();
        protected Optional<Long> randomSeed = Optional.empty();
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
        protected ScoringStrategy scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;
        protected boolean normalizeTime = false;
        protected double[] fillValues = null;
        protected double[] weights = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected boolean adjustThreshold = DEFAULT_AUTO_THRESHOLD;
        protected boolean learnNearIgnoreExpected = false;
        protected Optional<Double> transformDecay = Optional.empty();
        protected Optional<double[]> ignoreNearExpectedFromAbove = Optional.empty();
        protected Optional<double[]> ignoreNearExpectedFromBelow = Optional.empty();
        protected Optional<double[]> ignoreNearExpectedFromAboveByRatio = Optional.empty();
        protected Optional<double[]> ignoreNearExpectedFromBelowByRatio = Optional.empty();

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
                internalShinglingEnabled.ifPresent(x -> checkArgument(x,
                        " input cannot be shingled (even if internal representation is different) "));
            } else {
                if (!internalShinglingEnabled.isPresent()) {
                    internalShinglingEnabled = Optional.of(true);
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
                    .sampleSize(sampleSize).numberOfTrees(numberOfTrees).compact(true)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled).centerOfMassEnabled(centerOfMassEnabled)
                    .parallelExecutionEnabled(parallelExecutionEnabled).precision(precision)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).shingleSize(shingleSize)
                    .internalShinglingEnabled(internalShinglingEnabled.get())
                    .initialAcceptFraction(initialAcceptFraction);
            if (forestMode != ForestMode.STREAMING_IMPUTE) {
                outputAfter.ifPresent(builder::outputAfter);
            } else {
                // forcing the change between internal and external shingling
                outputAfter.ifPresent(n -> {
                    int num = max(startNormalization.orElse(Preprocessor.DEFAULT_START_NORMALIZATION), n) - shingleSize
                            + 1;
                    checkArgument(num > 0, " max(start normalization, output after) should be at least " + shingleSize);
                    builder.outputAfter(num);
                });
            }
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

        public T transformDecay(double transformDecay) {
            this.transformDecay = Optional.of(transformDecay);
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

        @Deprecated
        public T compact(boolean compact) {
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

        public T scoreDifferencing(double persistence) {
            this.scoreDifferencing = Optional.of(persistence);
            return (T) this;
        }

        public T adjustThreshold(boolean adjustThreshold) {
            this.adjustThreshold = adjustThreshold;
            return (T) this;
        }

        public T learnIgnoreNearExpected(boolean learnNearIgnoreExpected) {
            this.learnNearIgnoreExpected = learnNearIgnoreExpected;
            return (T) this;
        }

        public T weightTime(double value) {
            this.weightTime = Optional.of(value);
            return (T) this;
        }

        public T ignoreNearExpectedFromAbove(double[] ignoreSimilarFromAbove) {
            this.ignoreNearExpectedFromAbove = Optional.ofNullable(ignoreSimilarFromAbove);
            return (T) this;
        }

        public T ignoreNearExpectedFromBelow(double[] ignoreSimilarFromBelow) {
            this.ignoreNearExpectedFromBelow = Optional.ofNullable(ignoreSimilarFromBelow);
            return (T) this;
        }

        public T ignoreNearExpectedFromAboveByRatio(double[] ignoreSimilarFromAboveByRatio) {
            this.ignoreNearExpectedFromAboveByRatio = Optional.ofNullable(ignoreSimilarFromAboveByRatio);
            return (T) this;
        }

        public T ignoreNearExpectedFromBelowByRatio(double[] ignoreSimilarFromBelowByRatio) {
            this.ignoreNearExpectedFromBelowByRatio = Optional.ofNullable(ignoreSimilarFromBelowByRatio);
            return (T) this;
        }

        public T scoringStrategy(ScoringStrategy scoringStrategy) {
            this.scoringStrategy = scoringStrategy;
            return (T) this;
        }
    }
}
