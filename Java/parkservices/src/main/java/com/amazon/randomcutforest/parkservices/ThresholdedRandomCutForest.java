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
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_HORIZON;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_HORIZON_ONED;
import static com.amazon.randomcutforest.parkservices.threshold.BasicThresholder.DEFAULT_LOWER_THRESHOLD;
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

    protected RandomCutForest forest;

    protected PredictorCorrector predictorCorrector;

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
            checkArgument(builder.shingleSize > 1, " shingle size 1 is not useful in impute");
            preprocessorBuilder.inputLength(builder.dimensions / builder.shingleSize);

            preprocessorBuilder.imputationMethod(builder.imputationMethod);
            preprocessorBuilder.normalizeTime(true);
            if (builder.fillValues != null) {
                preprocessorBuilder.fillValues(builder.fillValues);
            }
            // forcing external for the forest to control admittance
            builder.internalShinglingEnabled = Optional.of(false);
            preprocessorBuilder.useImputedFraction(builder.useImputedFraction.orElse(0.5));
        } else {
            boolean smallInput = builder.internalShinglingEnabled.orElse(DEFAULT_INTERNAL_SHINGLING_ENABLED);
            preprocessorBuilder
                    .inputLength((smallInput) ? builder.dimensions / builder.shingleSize : builder.dimensions);
        }

        forest = builder.buildForest();
        preprocessorBuilder.weights(builder.weights);
        preprocessorBuilder.weightTime(builder.weightTime.orElse(1.0));
        preprocessorBuilder.timeDecay(forest.getTimeDecay());

        preprocessorBuilder.dimensions(builder.dimensions);
        preprocessorBuilder
                .stopNormalization(builder.stopNormalization.orElse(Preprocessor.DEFAULT_STOP_NORMALIZATION));
        preprocessorBuilder
                .startNormalization(builder.startNormalization.orElse(Preprocessor.DEFAULT_START_NORMALIZATION));

        preprocessor = preprocessorBuilder.build();
        predictorCorrector = new PredictorCorrector(new BasicThresholder(builder.anomalyRate, builder.adjustThreshold));
        lastAnomalyDescriptor = new RCFComputeDescriptor(null, 0, builder.forestMode, builder.transformMethod,
                builder.imputationMethod);

        // multiple (not extremely well correlated) dimensions typically reduce scores
        // normalization reduces scores
        if (preprocessor.getDimension() == preprocessor.getShingleSize()) {
            if (builder.transformMethod != TransformMethod.NORMALIZE) {
                predictorCorrector.setLowerThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD_ONED));
            } else {
                predictorCorrector.setLowerThreshold(
                        builder.lowerThreshold.orElse(BasicThresholder.DEFAULT_LOWER_THRESHOLD_NORMALIZED));
            }
            predictorCorrector.setHorizon(builder.horizon.orElse(DEFAULT_HORIZON_ONED));
        } else {
            if (builder.transformMethod != TransformMethod.NORMALIZE) {
                predictorCorrector.setLowerThreshold(builder.lowerThreshold.orElse(DEFAULT_LOWER_THRESHOLD));
            } else {
                predictorCorrector.setLowerThreshold(
                        builder.lowerThreshold.orElse(BasicThresholder.DEFAULT_LOWER_THRESHOLD_NORMALIZED));
            }
            predictorCorrector.setHorizon(builder.horizon.orElse(DEFAULT_HORIZON));
        }

    }

    // for mappers
    public ThresholdedRandomCutForest(ForestMode forestMode, TransformMethod transformMethod, RandomCutForest forest,
            PredictorCorrector predictorCorrector, Preprocessor preprocessor, RCFComputeDescriptor descriptor) {
        this.forestMode = forestMode;
        this.transformMethod = transformMethod;
        this.forest = forest;
        this.predictorCorrector = predictorCorrector;
        this.preprocessor = preprocessor;
        this.lastAnomalyDescriptor = descriptor;
    }

    // for conversion from other thresholding models
    public ThresholdedRandomCutForest(RandomCutForest forest, double futureAnomalyRate, List<Double> values) {
        this.forest = forest;
        this.predictorCorrector = new PredictorCorrector(new BasicThresholder(values, futureAnomalyRate));
        int dimensions = forest.getDimensions();
        int inputLength = (forest.isInternalShinglingEnabled()) ? dimensions / forest.getShingleSize()
                : forest.getDimensions();
        this.preprocessor = new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).dimensions(dimensions)
                .shingleSize(forest.getShingleSize()).inputLength(inputLength).build();
        preprocessor.setValuesSeen((int) forest.getTotalUpdates());
        preprocessor.getDataQuality().update(1.0);
        this.lastAnomalyDescriptor = new RCFComputeDescriptor(null, forest.getTotalUpdates());
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

        boolean ifZero = (forest.getBoundingBoxCacheFraction() == 0);
        if (ifZero) { // turn caching on temporarily
            forest.setBoundingBoxCacheFraction(1.0);
        }

        AnomalyDescriptor description = new AnomalyDescriptor(inputPoint, timestamp);

        preprocessor.preProcess(description, lastAnomalyDescriptor, forest);

        // score anomalies
        predictorCorrector.addAnomalyDescription(description, lastAnomalyDescriptor, forest);

        // add explanation
        preprocessor.postProcess(description, lastAnomalyDescriptor, forest);

        if (ifZero) { // turn caching off
            forest.setBoundingBoxCacheFraction(0);
        }

        if (description.getAnomalyGrade() > 0) {
            lastAnomalyDescriptor = description.copyOf();
        }
        return description;

    }

    public RandomCutForest getForest() {
        return forest;
    }

    public BasicThresholder getThresholder() {
        return predictorCorrector.getThresholder();
    }

    public void setZfactor(double factor) {
        predictorCorrector.setZfactor(factor);
    }

    public void setLowerThreshold(double lower) {
        predictorCorrector.setLowerThreshold(lower);
    }

    public void setHorizon(double horizon) {
        predictorCorrector.setHorizon(horizon);
    }

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
        protected Optional<Double> horizon = Optional.empty();
        protected Optional<Double> lowerThreshold = Optional.empty();
        protected Optional<Double> weightTime = Optional.empty();
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
                internalShinglingEnabled.ifPresent(x -> checkArgument(x,
                        " input cannot be shingled (even if internal representation is different) "));
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

        public T weightTime(double value) {
            this.weightTime = Optional.of(value);
            return (T) this;
        }

    }
}
