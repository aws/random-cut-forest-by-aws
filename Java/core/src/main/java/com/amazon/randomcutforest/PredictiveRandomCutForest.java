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

package com.amazon.randomcutforest;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INITIAL_ACCEPT_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_NUMBER_OF_TREES;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_OUTPUT_AFTER_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_PARALLEL_EXECUTION_ENABLED;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SHINGLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static com.amazon.randomcutforest.config.ImputationMethod.PREVIOUS;
import static com.amazon.randomcutforest.preprocessor.Preprocessor.DEFAULT_START_NORMALIZATION;
import static com.amazon.randomcutforest.preprocessor.Preprocessor.DEFAULT_STOP_NORMALIZATION;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.preprocessor.IPreprocessor;
import com.amazon.randomcutforest.preprocessor.Preprocessor;
import com.amazon.randomcutforest.returntypes.SampleSummary;

/**
 * This class provides a predictive imputation based on RCF (respecting the
 * arrow of time) alongside streaming normalization
 */
@Getter
@Setter
public class PredictiveRandomCutForest {

    protected TransformMethod transformMethod = TransformMethod.NORMALIZE;

    protected RandomCutForest forest;

    protected IPreprocessor preprocessor;

    protected ForestMode forestMode = ForestMode.STANDARD;

    public PredictiveRandomCutForest(Builder<?> builder) {
        checkArgument(builder.forestMode != ForestMode.STREAMING_IMPUTE, "not suppoted yet");
        transformMethod = builder.transformMethod;
        Preprocessor.Builder<?> preprocessorBuilder = Preprocessor.builder().shingleSize(builder.shingleSize)
                .transformMethod(builder.transformMethod).forestMode(builder.forestMode);

        int dimensions = builder.inputDimensions * builder.shingleSize;
        if (builder.forestMode == ForestMode.TIME_AUGMENTED) {
            dimensions += builder.shingleSize;
            preprocessorBuilder.normalizeTime(true);
            // force internal shingling for this option
            builder.internalShinglingEnabled = Optional.of(true);
        }

        forestMode = builder.forestMode;
        forest = builder.buildForest();
        validateNonNegativeArray(builder.weights);

        preprocessorBuilder.inputLength(builder.inputDimensions);
        preprocessorBuilder.weights(builder.weights);
        preprocessorBuilder.weightTime(builder.weightTime.orElse(1.0));
        preprocessorBuilder.timeDecay(builder.transformDecay.orElse(1.0 / builder.sampleSize));
        // to be used later
        preprocessorBuilder.randomSeed(builder.randomSeed.orElse(0L) + 1);
        preprocessorBuilder.dimensions(dimensions);
        preprocessorBuilder.stopNormalization(builder.stopNormalization.orElse(DEFAULT_STOP_NORMALIZATION));
        preprocessorBuilder.startNormalization(builder.startNormalization.orElse(DEFAULT_START_NORMALIZATION));

        preprocessor = preprocessorBuilder.build();
    }

    public PredictiveRandomCutForest(ForestMode forestMode, TransformMethod method, IPreprocessor preprocessor,
            RandomCutForest forest) {
        this.forestMode = forestMode;
        this.transformMethod = method;
        this.preprocessor = preprocessor;
        this.forest = forest;
    }

    void validateNonNegativeArray(double[] array) {
        if (array != null) {
            for (double element : array) {
                checkArgument(element >= 0, " has to be non-negative");
            }
        }
    }

    public SampleSummary predict(float[] inputPoint, long timestamp, int[] missingValues) {
        return predict(inputPoint, timestamp, missingValues, 5, 0.3, 0.7, false);
    }

    public SampleSummary predict(float[] inputPoint, long timestamp, int[] missingValues, int numberOfRepresentatives,
            double shrinkage, double centrality, boolean project) {

        SampleSummary summary = null;
        boolean cacheDisabled = (forest.getBoundingBoxCacheFraction() == 0);
        try {
            if (cacheDisabled) { // turn caching on temporarily
                forest.setBoundingBoxCacheFraction(1.0);
            }
            if (missingValues != null) {
                checkArgument(missingValues.length <= inputPoint.length, " incorrect data");
                for (int i = 0; i < missingValues.length; i++) {
                    checkArgument(missingValues[i] >= 0, " missing values cannot be at negative position");
                    checkArgument(missingValues[i] < inputPoint.length,
                            "missing values cannot be at position larger than input length");
                }
            }
            float[] scaledInput = preprocessor.getScaledInput(inputPoint, timestamp);
            summary = preprocessor.invertSummary(
                    forest.getConditionalFieldSummary(scaledInput, missingValues.length, missingValues,
                            numberOfRepresentatives, shrinkage, true, project, centrality),
                    missingValues.length, missingValues, scaledInput);
        } finally {
            if (cacheDisabled) { // turn caching off
                forest.setBoundingBoxCacheFraction(0);
            }
        }
        return summary;
    }

    public void update(float[] record, long timestamp) {
        float[] scaled = preprocessor.getScaledInput(record, timestamp);
        preprocessor.update(toDoubleArray(record), scaled, timestamp, null, forest);
        if (scaled != null) {
            forest.update(scaled);
        }
    }

    public RandomCutForest getForest() {
        return forest;
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

        protected int inputDimensions;
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
        protected double boundingBoxCacheFraction = DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected int shingleSize = DEFAULT_SHINGLE_SIZE;
        protected Optional<Boolean> internalShinglingEnabled = Optional.empty();
        protected double initialAcceptFraction = DEFAULT_INITIAL_ACCEPT_FRACTION;
        protected double anomalyRate = 0.01;
        protected TransformMethod transformMethod = TransformMethod.NONE;
        protected ImputationMethod imputationMethod = PREVIOUS;
        protected ForestMode forestMode = ForestMode.STANDARD;
        protected double[] weights = null;
        protected Optional<Double> useImputedFraction = Optional.empty();
        protected Optional<Double> transformDecay = Optional.empty();

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

        public PredictiveRandomCutForest build() {
            validate();
            return new PredictiveRandomCutForest(this);
        }

        protected RandomCutForest buildForest() {
            int dimensions = inputDimensions * shingleSize
                    + ((forestMode == ForestMode.TIME_AUGMENTED) ? shingleSize : 0);
            RandomCutForest.Builder builder = new RandomCutForest.Builder().dimensions(dimensions)
                    .sampleSize(sampleSize).numberOfTrees(numberOfTrees)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled).centerOfMassEnabled(centerOfMassEnabled)
                    .parallelExecutionEnabled(parallelExecutionEnabled)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).shingleSize(shingleSize)
                    .internalShinglingEnabled(internalShinglingEnabled.get())
                    .initialAcceptFraction(initialAcceptFraction);

            outputAfter.ifPresent(builder::outputAfter);
            timeDecay.ifPresent(builder::timeDecay);
            randomSeed.ifPresent(builder::randomSeed);
            threadPoolSize.ifPresent(builder::threadPoolSize);
            return builder.build();
        }

        public T inputDimensions(int dimensions) {
            this.inputDimensions = dimensions;
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

        public T forestMode(ForestMode forestMode) {
            this.forestMode = forestMode;
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

        public T internalShinglingEnabled(boolean internalShinglingEnabled) {
            this.internalShinglingEnabled = Optional.of(internalShinglingEnabled);
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

        public T weights(double[] values) {
            // values cannot be a null
            this.weights = Arrays.copyOf(values, values.length);
            return (T) this;
        }

        public T transformMethod(TransformMethod method) {
            this.transformMethod = method;
            return (T) this;
        }
    }
}
