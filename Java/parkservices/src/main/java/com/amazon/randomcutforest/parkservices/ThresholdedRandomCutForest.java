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
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_OUTPUT_AFTER_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PARALLEL_EXECUTION_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PRECISION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor.DEFAULT_START_NORMALIZATION;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_ABSOLUTE_THRESHOLD;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_AUTO_THRESHOLD;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_SCORE_DIFFERENCING;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_Z_FACTOR;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
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
    RCFComputeDescriptor lastAnomalyDescriptor;

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
        validateNonNegativeArray(builder.weights);

        preprocessorBuilder.weights(builder.weights);
        preprocessorBuilder.weightTime(builder.weightTime.orElse(1.0));
        preprocessorBuilder.timeDecay(builder.transformDecay.orElse(1.0 / builder.sampleSize));
        // to be used later
        preprocessorBuilder.randomSeed(builder.randomSeed.orElse(0L) + 1);
        preprocessorBuilder.dimensions(builder.dimensions);
        preprocessorBuilder
                .stopNormalization(builder.stopNormalization.orElse(Preprocessor.DEFAULT_STOP_NORMALIZATION));
        preprocessorBuilder.startNormalization(builder.startNormalization.orElse(DEFAULT_START_NORMALIZATION));

        preprocessor = preprocessorBuilder.build();
        predictorCorrector = new PredictorCorrector(forest.getTimeDecay(), builder.anomalyRate, builder.adjustThreshold,
                builder.learnIgnoreNearExpected, builder.dimensions / builder.shingleSize,
                builder.randomSeed.orElse(0L));
        lastAnomalyDescriptor = new RCFComputeDescriptor(null, 0, builder.forestMode, builder.transformMethod,
                builder.imputationMethod);

        predictorCorrector.setAbsoluteThreshold(builder.lowerThreshold.orElse(DEFAULT_ABSOLUTE_THRESHOLD));
        predictorCorrector.setZfactor(builder.zFactor);

        predictorCorrector.setScoreDifferencing(builder.scoreDifferencing.orElse(DEFAULT_SCORE_DIFFERENCING));
        builder.ignoreNearExpectedFromAbove.ifPresent(predictorCorrector::setIgnoreNearExpectedFromAbove);
        builder.ignoreNearExpectedFromBelow.ifPresent(predictorCorrector::setIgnoreNearExpectedFromBelow);
        builder.ignoreNearExpectedFromAboveByRatio.ifPresent(predictorCorrector::setIgnoreNearExpectedFromAboveByRatio);
        builder.ignoreNearExpectedFromBelowByRatio.ifPresent(predictorCorrector::setIgnoreNearExpectedFromBelowByRatio);
        predictorCorrector.setLastStrategy(builder.scoringStrategy);
        predictorCorrector.setIgnoreDrift(builder.alertOnceInDrift);
    }

    void validateNonNegativeArray(double[] array) {
        if (array != null) {
            for (double element : array) {
                checkArgument(element >= 0, " has to be non-negative");
            }
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

    protected <T extends AnomalyDescriptor> boolean saveDescriptor(T lastDescriptor) {
        return (lastDescriptor.getAnomalyGrade() > 0);
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
        return preprocessor.postProcess(core.apply(preprocessor.preProcess(input, lastAnomalyDescriptor, forest)),
                lastAnomalyDescriptor, forest);
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

        AnomalyDescriptor description = new AnomalyDescriptor(inputPoint, timestamp);
        description.setScoringStrategy(scoringStrategy);
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        try {
            if (cacheDisabled) { // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            if (missingValues != null) {
                checkArgument(missingValues.length <= inputPoint.length, " incorrect data");
                for (int i = 0; i < missingValues.length; i++) {
                    checkArgument(missingValues[i] >= 0 && missingValues[i] < inputPoint.length,
                            " incorrect positions ");
                }
                description.setMissingValues(missingValues);
            }
            description = singleStepProcess(description, preprocessor, function);
        } finally {
            if (cacheDisabled) { // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }
        if (saveDescriptor(description)) {
            lastAnomalyDescriptor = description.copyOf();
        }
        return description;
    }

    /**
     * the following function processes a list of vectors sequentially; the main
     * benefit of this invocation is the caching is persisted from one data point to
     * another and thus the execution is efficient. Moreover in many scenarios where
     * serialization deserialization is expensive then it may be of benefit of
     * invoking sequential process on a contiguous chunk of input (we avoid the use
     * of the word batch -- the entire goal of this procedure is to provide
     * sequential processing and not standard batch processing). The procedure
     * avoids transfer of ephemeral transient objects for non-anomalies and thereby
     * can have additional benefits. At the moment the operation does not support
     * external timestamps.
     *
     * @param data   a vectors of vectors (each of which has to have the same
     *               inputLength)
     * @param filter a condition to drop desriptor (recommended filter: anomalyGrade
     *               positive)
     * @return collection of descriptors of the anomalies filtered by the condition
     */
    public List<AnomalyDescriptor> processSequentially(double[][] data, Function<AnomalyDescriptor, Boolean> filter) {
        ArrayList<AnomalyDescriptor> answer = new ArrayList<>();
        Function<AnomalyDescriptor, AnomalyDescriptor> function = (x) -> predictorCorrector.detect(x,
                lastAnomalyDescriptor, forest);
        if (data != null && data.length > 0) {
            boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
            try {
                if (cacheDisabled) { // turn caching on temporarily
                    forest.setBoundingBoxCacheFraction(1.0);
                }
                long timestamp = preprocessor.getInternalTimeStamp();
                int length = preprocessor.getInputLength();
                for (double[] point : data) {
                    checkArgument(point.length == length, " nonuniform lengths ");
                    AnomalyDescriptor description = new AnomalyDescriptor(point, timestamp++);
                    description.setScoringStrategy(scoringStrategy);
                    description = singleStepProcess(description, preprocessor, function);
                    if (saveDescriptor(description)) {
                        lastAnomalyDescriptor = description.copyOf();
                    }
                    if (filter.apply(description)) {
                        answer.add(description);
                    }
                }
            } finally {
                if (cacheDisabled) { // turn caching off
                    forest.setBoundingBoxCacheFraction(0);
                }
            }
        }
        return answer;
    }

    // recommended filter
    public List<AnomalyDescriptor> processSequentially(double[][] data) {
        return processSequentially(data, x -> x.anomalyGrade > 0);
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

        int shingleSize = preprocessor.getShingleSize();
        checkArgument(shingleSize > 1, "extrapolation is not meaningful for shingle size = 1");
        // note the forest may have external shingling ...
        int dimensions = forest.getDimensions();
        int blockSize = dimensions / shingleSize;
        double[] lastPoint = preprocessor.getLastShingledPoint();
        if (forest.isOutputReady()) {
            int gap = (int) (preprocessor.getInternalTimeStamp() - lastAnomalyDescriptor.getInternalTimeStamp());

            float[] newPoint = toFloatArray(lastPoint);

            // gap will be at least 1
            if (gap <= shingleSize && correct && lastAnomalyDescriptor.getExpectedRCFPoint() != null) {
                if (gap == 1) {
                    newPoint = toFloatArray(lastAnomalyDescriptor.getExpectedRCFPoint());
                } else {
                    newPoint = predictorCorrector.applyPastCorrector(newPoint, gap, shingleSize, blockSize,
                            preprocessor.getScale(), transformMethod, lastAnomalyDescriptor);
                }
            }
            RangeVector answer = forest.extrapolateFromShingle(newPoint, horizon, blockSize, centrality);
            return preprocessor.invertForecastRange(answer, lastAnomalyDescriptor);
        } else {
            return new TimedRangeVector(new TimedRangeVector(horizon * blockSize, horizon));
        }
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
        predictorCorrector.setAbsoluteThreshold(lower);
    }

    @Deprecated
    public void setHorizon(double horizon) {
        predictorCorrector.setScoreDifferencing(1 - horizon);
    }

    public void setScoreDifferencing(double scoreDifferencing) {
        predictorCorrector.setScoreDifferencing(scoreDifferencing);
    }

    public void setScoringStrategy(ScoringStrategy strategy) {
        this.scoringStrategy = strategy;
    }

    @Deprecated
    public void setInitialThreshold(double initial) {
        predictorCorrector.setInitialThreshold(initial);
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
        protected boolean learnIgnoreNearExpected = false;
        protected double zFactor = DEFAULT_Z_FACTOR;
        protected boolean alertOnceInDrift = false;
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
            if (startNormalization.isPresent()) {
                // we should not be setting normalizations unless we are careful
                if (outputAfter.isPresent()) {
                    // can be overspecified
                    checkArgument(outputAfter.get() + shingleSize - 1 > startNormalization.get(),
                            "output after has to wait till normalization, reduce normalization");
                } else {
                    int n = startNormalization.get();
                    checkArgument(n > 0, " startNormalization has to be positive");
                    // if start normalization is low then first few output can be 0
                    outputAfter = Optional
                            .of(max(max(1, (int) (sampleSize * DEFAULT_OUTPUT_AFTER_FRACTION)), n - shingleSize + 1));
                }
            } else {
                if (outputAfter.isPresent()) {
                    startNormalization = Optional.of(min(DEFAULT_START_NORMALIZATION, outputAfter.get()));
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
                    int num = max(startNormalization.orElse(DEFAULT_START_NORMALIZATION), n) - shingleSize + 1;
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

        public T zFactor(double zFactor) {
            this.zFactor = zFactor;
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
            this.learnIgnoreNearExpected = learnNearIgnoreExpected;
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

        public T alertOnce(boolean alertOnceInDrift) {
            this.alertOnceInDrift = alertOnceInDrift;
            return (T) this;
        }
    }
}
