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
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static java.lang.Math.max;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import com.amazon.randomcutforest.anomalydetection.AnomalyAttributionVisitor;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.anomalydetection.DynamicAttributionVisitor;
import com.amazon.randomcutforest.anomalydetection.DynamicScoreVisitor;
import com.amazon.randomcutforest.anomalydetection.SimulatedTransductiveScalarScoreVisitor;
import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.executor.AbstractForestUpdateExecutor;
import com.amazon.randomcutforest.executor.IStateCoordinator;
import com.amazon.randomcutforest.executor.ParallelForestTraversalExecutor;
import com.amazon.randomcutforest.executor.ParallelForestUpdateExecutor;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.executor.SequentialForestTraversalExecutor;
import com.amazon.randomcutforest.executor.SequentialForestUpdateExecutor;
import com.amazon.randomcutforest.imputation.ImputeVisitor;
import com.amazon.randomcutforest.inspect.NearNeighborVisitor;
import com.amazon.randomcutforest.interpolation.SimpleInterpolationVisitor;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDiVectorAccumulator;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDoubleAccumulator;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.sampler.IStreamSampler;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.tree.IBoundingBoxView;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.ArrayUtils;
import com.amazon.randomcutforest.util.ShingleBuilder;

/**
 * The RandomCutForest class is the interface to the algorithms in this package,
 * and includes methods for anomaly detection, anomaly detection with
 * attribution, density estimation, imputation, and forecasting. A Random Cut
 * Forest is a collection of Random Cut Trees and stream samplers. When an
 * update call is made to a Random Cut Forest, each sampler is independently
 * updated with the submitted (and if the point is accepted by the sampler, then
 * the corresponding Random Cut Tree is also updated. Similarly, when an
 * algorithm method is called, the Random Cut Forest proxies to the trees which
 * implement the actual scoring logic. The Random Cut Forest then combines
 * partial results into a final results.
 */
public class RandomCutForest {

    /**
     * Default sample size. This is the number of points retained by the stream
     * sampler.
     */
    public static final int DEFAULT_SAMPLE_SIZE = 256;

    /**
     * Default fraction used to compute the amount of points required by stream
     * samplers before results are returned.
     */
    public static final double DEFAULT_OUTPUT_AFTER_FRACTION = 0.25;

    /**
     * If the user doesn't specify an explicit time decay value, then we set it to
     * the inverse of this coefficient times sample size.
     */
    public static final double DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY = 10.0;

    /**
     * Default number of trees to use in the forest.
     */
    public static final int DEFAULT_NUMBER_OF_TREES = 50;

    /**
     * By default, trees will not store sequence indexes.
     */
    public static final boolean DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED = false;

    /**
     * By default, trees will accept every point until full.
     */
    public static final double DEFAULT_INITIAL_ACCEPT_FRACTION = 1.0;

    /**
     * By default, the collection of points stored in the forest will increase from
     * a small size, as needed to maximum capacity
     */
    public static final boolean DEFAULT_DYNAMIC_RESIZING_ENABLED = true;

    /**
     * By default, shingling will be external
     */
    public static final boolean DEFAULT_INTERNAL_SHINGLING_ENABLED = false;

    /**
     * By default, shingles will be a sliding window and not a cyclic buffer
     */
    public static final boolean DEFAULT_INTERNAL_ROTATION_ENABLED = false;

    /**
     * By default, point stores will favor speed of size for larger shingle sizes
     */
    public static final boolean DEFAULT_DIRECT_LOCATION_MAP = false;

    /**
     * Default floating-point precision for internal data structures.
     */
    public static final Precision DEFAULT_PRECISION = Precision.FLOAT_64;

    /**
     * fraction of bounding boxes maintained by each tree
     */
    public static final double DEFAULT_BOUNDING_BOX_CACHE_FRACTION = 1.0;

    /**
     * By default, nodes will not store center of mass.
     */
    public static final boolean DEFAULT_CENTER_OF_MASS_ENABLED = false;

    /**
     * By default RCF is unaware of shingle size
     */
    public static final int DEFAULT_SHINGLE_SIZE = 1;

    /**
     * Parallel execution is enabled by default.
     */
    public static final boolean DEFAULT_PARALLEL_EXECUTION_ENABLED = false;

    public static final boolean DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL = true;

    public static final double DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION = 0.1;

    public static final int DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED = 5;

    /**
     * Random number generator used by the forest.
     */
    protected Random random;
    /**
     * The number of dimensions in the input data.
     */
    protected final int dimensions;
    /**
     * The sample size used by stream samplers in this forest.
     */
    protected final int sampleSize;
    /**
     * The shingle size (if known)
     */
    protected final int shingleSize;
    /**
     * The input dimensions for known shingle size and internal shingling
     */
    protected final int inputDimensions;
    /**
     * The number of points required by stream samplers before results are returned.
     */
    protected final int outputAfter;
    /**
     * The number of trees in this forest.
     */
    protected final int numberOfTrees;
    /**
     * The decay factor used by stream samplers in this forest.
     */
    protected double timeDecay;
    /**
     * Store the time information
     */
    protected final boolean storeSequenceIndexesEnabled;

    /**
     * enables internal shingling
     */
    protected final boolean internalShinglingEnabled;

    /**
     * The following can be set between 0 and 1 (inclusive) to achieve tradeoff
     * between smaller space, lower throughput and larger space, larger throughput
     */
    protected final double boundingBoxCacheFraction;
    /**
     * Enable center of mass at internal nodes
     */
    protected final boolean centerOfMassEnabled;
    /**
     * Enable parallel execution.
     */
    protected final boolean parallelExecutionEnabled;
    /**
     * Number of threads to use in the thread pool if parallel execution is enabled.
     */
    protected final int threadPoolSize;
    /**
     * A string to define an "execution mode" that can be used to set multiple
     * configuration options. This field is not currently in use.
     */
    protected String executionMode;

    protected IStateCoordinator<?, ?> stateCoordinator;
    protected ComponentList<?, ?> components;

    /**
     * This flag is initialized to false. It is set to true when all component
     * models are ready.
     */
    private boolean outputReady;

    /**
     * used for initializing the compact forests
     */
    private final int initialPointStoreSize;
    private final int pointStoreCapacity;

    /**
     * An implementation of forest traversal algorithms.
     */
    protected AbstractForestTraversalExecutor traversalExecutor;

    /**
     * An implementation of forest update algorithms.
     */
    protected AbstractForestUpdateExecutor<?, ?> updateExecutor;

    public <P, Q> RandomCutForest(Builder<?> builder, IStateCoordinator<P, Q> stateCoordinator,
            ComponentList<P, Q> components, Random random) {
        this(builder, false);

        checkNotNull(stateCoordinator, "updateCoordinator must not be null");
        checkNotNull(components, "componentModels must not be null");
        checkNotNull(random, "random must not be null");

        this.stateCoordinator = stateCoordinator;
        this.components = components;
        this.random = random;
        initExecutors(stateCoordinator, components);
    }

    public RandomCutForest(Builder<?> builder) {
        this(builder, false);
        random = builder.getRandom();

        PointStore tempStore = PointStore.builder().internalRotationEnabled(builder.internalRotationEnabled)
                .capacity(pointStoreCapacity).initialSize(initialPointStoreSize)
                .directLocationEnabled(builder.directLocationMapEnabled)
                .internalShinglingEnabled(internalShinglingEnabled)
                .dynamicResizingEnabled(builder.dynamicResizingEnabled).shingleSize(shingleSize).dimensions(dimensions)
                .build();

        IStateCoordinator<Integer, float[]> stateCoordinator = new PointStoreCoordinator<>(tempStore);
        ComponentList<Integer, float[]> components = new ComponentList<>(numberOfTrees);
        for (int i = 0; i < numberOfTrees; i++) {
            ITree<Integer, float[]> tree = new RandomCutTree.Builder().capacity(sampleSize)
                    .randomSeed(random.nextLong()).pointStoreView(tempStore)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).centerOfMassEnabled(centerOfMassEnabled)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled).outputAfter(outputAfter).build();

            IStreamSampler<Integer> sampler = CompactSampler.builder().capacity(sampleSize).timeDecay(timeDecay)
                    .randomSeed(random.nextLong()).storeSequenceIndexesEnabled(storeSequenceIndexesEnabled)
                    .initialAcceptFraction(builder.initialAcceptFraction).build();

            components.add(new SamplerPlusTree<>(sampler, tree));
        }
        this.stateCoordinator = stateCoordinator;
        this.components = components;
        initExecutors(stateCoordinator, components);
    }

    protected <PointReference, Point> void initExecutors(IStateCoordinator<PointReference, Point> updateCoordinator,
            ComponentList<PointReference, Point> components) {
        if (parallelExecutionEnabled) {
            traversalExecutor = new ParallelForestTraversalExecutor(components, threadPoolSize);
            updateExecutor = new ParallelForestUpdateExecutor<>(updateCoordinator, components, threadPoolSize);
        } else {
            traversalExecutor = new SequentialForestTraversalExecutor(components);
            updateExecutor = new SequentialForestUpdateExecutor<>(updateCoordinator, components);
        }
    }

    /**
     * This constructor is responsible for initializing a forest's configuration
     * variables from a builder. The method signature contains a boolean argument
     * that isn't used. This argument exists only to create a distinct method
     * signature so that we can expose {@link #RandomCutForest(Builder)} as a
     * protected constructor.
     * 
     * @param builder A Builder instance giving the desired random cut forest
     *                configuration.
     * @param notUsed This parameter is not used.
     */
    protected RandomCutForest(Builder<?> builder, boolean notUsed) {
        checkArgument(builder.numberOfTrees > 0, "numberOfTrees must be greater than 0");
        checkArgument(builder.sampleSize > 0, "sampleSize must be greater than 0");
        builder.outputAfter.ifPresent(n -> {
            checkArgument(n > 0, "outputAfter must be greater than 0");
            checkArgument(n <= builder.sampleSize, "outputAfter must be smaller or equal to sampleSize");
        });
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        builder.timeDecay.ifPresent(timeDecay -> {
            checkArgument(timeDecay >= 0, "timeDecay must be greater than or equal to 0");
        });
        builder.threadPoolSize.ifPresent(n -> checkArgument((n > 0) || ((n == 0) && !builder.parallelExecutionEnabled),
                "threadPoolSize must be greater/equal than 0. To disable thread pool, set parallel execution to 'false'."));
        checkArgument(builder.internalShinglingEnabled || builder.shingleSize == 1
                || builder.dimensions % builder.shingleSize == 0, "wrong shingle size");
        // checkArgument(!builder.internalShinglingEnabled || builder.shingleSize > 1,
        // " need shingle size > 1 for internal shingling");
        if (builder.internalRotationEnabled) {
            checkArgument(builder.internalShinglingEnabled, " enable internal shingling");
        }
        builder.initialPointStoreSize.ifPresent(n -> {
            checkArgument(n > 0, "initial point store must be greater than 0");
            checkArgument(n > builder.sampleSize * builder.numberOfTrees || builder.dynamicResizingEnabled,
                    " enable dynamic resizing ");
        });
        checkArgument(builder.boundingBoxCacheFraction >= 0 && builder.boundingBoxCacheFraction <= 1,
                "incorrect cache fraction range");
        numberOfTrees = builder.numberOfTrees;
        sampleSize = builder.sampleSize;
        outputAfter = builder.outputAfter.orElse((int) (sampleSize * DEFAULT_OUTPUT_AFTER_FRACTION));
        internalShinglingEnabled = builder.internalShinglingEnabled;
        shingleSize = builder.shingleSize;
        dimensions = builder.dimensions;
        timeDecay = builder.timeDecay.orElse(1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY * sampleSize));
        storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        centerOfMassEnabled = builder.centerOfMassEnabled;
        parallelExecutionEnabled = builder.parallelExecutionEnabled;
        boundingBoxCacheFraction = builder.boundingBoxCacheFraction;
        builder.directLocationMapEnabled = builder.directLocationMapEnabled || shingleSize == 1;
        inputDimensions = (internalShinglingEnabled) ? dimensions / shingleSize : dimensions;
        pointStoreCapacity = max(sampleSize * numberOfTrees + 1, 2 * sampleSize);
        initialPointStoreSize = builder.initialPointStoreSize.orElse(2 * sampleSize);

        if (parallelExecutionEnabled) {
            threadPoolSize = builder.threadPoolSize.orElse(Runtime.getRuntime().availableProcessors() - 1);
        } else {
            threadPoolSize = 0;
        }
    }

    /**
     * @return a new RandomCutForest builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a new RandomCutForest with optional arguments set to default values.
     *
     * @param dimensions The number of dimension in the input data.
     * @param randomSeed The random seed to use to create the forest random number
     *                   generator
     * @return a new RandomCutForest with optional arguments set to default values.
     */
    public static RandomCutForest defaultForest(int dimensions, long randomSeed) {
        return builder().dimensions(dimensions).randomSeed(randomSeed).build();
    }

    /**
     * Create a new RandomCutForest with optional arguments set to default values.
     *
     * @param dimensions The number of dimension in the input data.
     * @return a new RandomCutForest with optional arguments set to default values.
     */
    public static RandomCutForest defaultForest(int dimensions) {
        return builder().dimensions(dimensions).build();
    }

    /**
     * @return the number of trees in the forest.
     */
    public int getNumberOfTrees() {
        return numberOfTrees;
    }

    /**
     * @return the sample size used by stream samplers in this forest.
     */
    public int getSampleSize() {
        return sampleSize;
    }

    /**
     * @return the shingle size used by the point store.
     */
    public int getShingleSize() {
        return shingleSize;
    }

    /**
     * @return the number of points required by stream samplers before results are
     *         returned.
     */
    public int getOutputAfter() {
        return outputAfter;
    }

    /**
     * @return the number of dimensions in the data points accepted by this forest.
     */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * @return return the decay factor used by stream samplers in this forest.
     */
    public double getTimeDecay() {
        return timeDecay;
    }

    /**
     * @return true if points are saved with sequence indexes, false otherwise.
     */
    public boolean isStoreSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    /**
     * For compact forests, users can choose to specify the desired floating-point
     * precision to use internally to store points. Choosing single-precision will
     * reduce the memory size of the model at the cost of requiring double/float
     * conversions.
     *
     * @return the desired precision to use internally to store points.
     */
    public Precision getPrecision() {
        return Precision.FLOAT_32;
    }

    @Deprecated
    public boolean isCompact() {
        return true;
    }

    /**
     * @return true if internal shingling is performed, false otherwise.
     */
    public boolean isInternalShinglingEnabled() {
        return internalShinglingEnabled;
    }

    /**
     * @return true if tree nodes retain the center of mass, false otherwise.
     */
    public boolean isCenterOfMassEnabled() {
        return centerOfMassEnabled;
    }

    /**
     * @return true if parallel execution is enabled, false otherwise.
     */
    public boolean isParallelExecutionEnabled() {
        return parallelExecutionEnabled;
    }

    public double getBoundingBoxCacheFraction() {
        return boundingBoxCacheFraction;
    }

    /**
     * @return the number of threads in the thread pool if parallel execution is
     *         enabled, 0 otherwise.
     */
    public int getThreadPoolSize() {
        return threadPoolSize;
    }

    public IStateCoordinator<?, ?> getUpdateCoordinator() {
        return stateCoordinator;
    }

    public ComponentList<?, ?> getComponents() {
        return components;
    }

    /**
     * used for scoring and other function, expands to a shingled point in either
     * case performs a clean copy
     * 
     * @param point input point
     * @return a shingled copy or a clean copy
     */
    public double[] transformToShingledPoint(double[] point) {
        checkNotNull(point, "point must not be null");
        return (internalShinglingEnabled && point.length == inputDimensions)
                ? toDoubleArray(stateCoordinator.getStore().transformToShingledPoint(toFloatArray(point)))
                : ArrayUtils.cleanCopy(point);
    }

    public float[] transformToShingledPoint(float[] point) {
        checkNotNull(point, "point must not be null");
        return (internalShinglingEnabled && point.length == inputDimensions)
                ? stateCoordinator.getStore().transformToShingledPoint(point)
                : ArrayUtils.cleanCopy(point);
    }

    /**
     * does the pointstore use rotated shingles
     * 
     * @return true/false based on pointstore
     */
    public boolean isRotationEnabled() {
        return stateCoordinator.getStore().isInternalRotationEnabled();
    }

    /**
     * transforms the missing indices on the input point to the corresponding
     * indices of a shingled point
     * 
     * @param indexList input array of missing values
     * @param length    length of the input array
     * @return output array of missing values corresponding to shingle
     */
    protected int[] transformIndices(int[] indexList, int length) {
        return (internalShinglingEnabled && length == inputDimensions)
                ? stateCoordinator.getStore().transformIndices(indexList)
                : indexList;
    }

    /**
     *
     * @return the last known shingled point seen
     */
    public float[] lastShingledPoint() {
        checkArgument(internalShinglingEnabled, "incorrect use");
        return stateCoordinator.getStore().getInternalShingle();
    }

    /**
     *
     * @return the sequence index of the last known shingled point
     */
    public long nextSequenceIndex() {
        checkArgument(internalShinglingEnabled, "incorrect use");
        return stateCoordinator.getStore().getNextSequenceIndex();
    }

    /**
     * Update the forest with the given point. The point is submitted to each
     * sampler in the forest. If the sampler accepts the point, the point is
     * submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        update(toFloatArray(point));
    }

    public void update(float[] point) {
        checkNotNull(point, "point must not be null");
        checkArgument(internalShinglingEnabled || point.length == dimensions,
                String.format("point.length must equal %d", dimensions));
        checkArgument(!internalShinglingEnabled || point.length == inputDimensions,
                String.format("point.length must equal %d for internal shingling", inputDimensions));

        updateExecutor.update(point);
    }

    /**
     * Update the forest with the given point and a timestamp. The point is
     * submitted to each sampler in the forest as if that timestamp was the correct
     * stamp. storeSequenceIndexes must be false since the algorithm will not verify
     * the correctness of the timestamp.
     *
     * @param point       The point used to update the forest.
     * @param sequenceNum The timestamp of the corresponding point
     */
    public void update(double[] point, long sequenceNum) {
        update(toFloatArray(point), sequenceNum);
    }

    public void update(float[] point, long sequenceNum) {
        checkNotNull(point, "point must not be null");
        checkArgument(!internalShinglingEnabled, "cannot be applied with internal shingling");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        updateExecutor.update(point, sequenceNum);
    }

    /**
     * Update the forest such that each tree caches a fraction of the bounding
     * boxes. This allows for a tradeoff between speed and storage.
     *
     * @param cacheFraction The (approximate) fraction of bounding boxes used in
     *                      caching.
     */
    public void setBoundingBoxCacheFraction(double cacheFraction) {
        checkArgument(0 <= cacheFraction && cacheFraction <= 1, "cacheFraction must be between 0 and 1 (inclusive)");
        updateExecutor.getComponents().forEach(c -> c.setConfig(Config.BOUNDING_BOX_CACHE_FRACTION, cacheFraction));
    }

    /**
     * changes the setting of time dependent sampling on the fly
     * 
     * @param timeDecay new value of sampling rate
     */
    public void setTimeDecay(double timeDecay) {
        checkArgument(0 <= timeDecay, "timeDecay must be greater than or equal to 0");
        this.timeDecay = timeDecay;
        updateExecutor.getComponents().forEach(c -> c.setConfig(Config.TIME_DECAY, timeDecay));
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to
     * {@link RandomCutTree#traverse(float[], IVisitorFactory)}. The results from
     * all the trees are combined using the accumulator and then transformed using
     * the finisher before being returned. Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return traversalExecutor.traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to
     * {@link RandomCutTree#traverse(float[], IVisitorFactory)}. The results from
     * individual trees are collected using the {@link java.util.stream.Collector}
     * and returned. Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, Collector<R, ?, S> collector) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(collector, "collector must not be null");

        return traversalExecutor.traverseForest(point, visitorFactory, collector);
    }

    /**
     * Visit each of the trees in the forest sequentially and combine the individual
     * results into an aggregate result. A visitor is constructed for each tree
     * using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverse(float[], IVisitorFactory)}. The results from
     * all the trees are combined using the {@link ConvergingAccumulator}, and the
     * method stops visiting trees after convergence is reached. The result is
     * transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    An accumulator that combines the results from
     *                       individual trees into an aggregate result and checks to
     *                       see if the result can be returned without further
     *                       processing.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return traversalExecutor.traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to
     * {@link RandomCutTree#traverseMulti(float[], IMultiVisitorFactory)}. The
     * results from all the trees are combined using the accumulator and then
     * transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a multi-visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return traversalExecutor.traverseForestMulti(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to
     * {@link RandomCutTree#traverseMulti(float[], IMultiVisitorFactory)}. The
     * results from individual trees are collected using the
     * {@link java.util.stream.Collector} and returned. Trees are visited in
     * parallel using {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(collector, "collector must not be null");

        return traversalExecutor.traverseForestMulti(point, visitorFactory, collector);
    }

    /**
     * Compute an anomaly score for the given point. The point being scored is
     * compared with the points in the sample to compute a measure of how anomalous
     * it is. Scores are greater than 0, with higher scores corresponding to bing
     * more anomalous. A threshold of 1.0 is commonly used to distinguish anomalous
     * points from non-anomalous ones.
     * <p>
     * See {@link AnomalyScoreVisitor} for more details about the anomaly score
     * algorithm.
     *
     * @param point The point being scored.
     * @return an anomaly score for the given point.
     */
    @Deprecated
    public double getAnomalyScore(double[] point) {
        return getAnomalyScore(toFloatArray(point));
    }

    public double getAnomalyScore(float[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        IVisitorFactory<Double> visitorFactory = (tree, x) -> new AnomalyScoreVisitor(tree.projectToTree(x),
                tree.getMass());
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Anomaly score evaluated sequentially with option of early stopping the early
     * stopping parameter precision gives an approximate solution in the range
     * (1-precision)*score(q)- precision, (1+precision)*score(q) + precision for the
     * score of a point q. In this function z is hardcoded to 0.1. If this function
     * is used, then not all the trees will be used in evaluation (but they have to
     * be updated anyways, because they may be used for the next q). The advantage
     * is that "almost certainly" anomalies/non-anomalies can be detected easily
     * with few trees.
     *
     * @param point input point q
     * @return anomaly score with early stopping with z=0.1
     */
    @Deprecated
    public double getApproximateAnomalyScore(double[] point) {
        return getApproximateAnomalyScore(toFloatArray(point));
    }

    public double getApproximateAnomalyScore(float[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        IVisitorFactory<Double> visitorFactory = (tree, x) -> new AnomalyScoreVisitor(tree.projectToTree(x),
                tree.getMass());

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Compute an anomaly score attribution DiVector for the given point. The point
     * being scored is compared with the points in the sample to compute a measure
     * of how anomalous it is. The result DiVector will contain an anomaly score in
     * both the positive and negative directions for each dimension of the data.
     * <p>
     * See {@link AnomalyAttributionVisitor} for more details about the anomaly
     * score algorithm.
     *
     * @param point The point being scored.
     * @return an anomaly score for the given point.
     */
    public DiVector getAnomalyAttribution(double[] point) {
        return getAnomalyAttribution(toFloatArray(point));
    }

    public DiVector getAnomalyAttribution(float[] point) {
        // this will return the same (modulo floating point summation) L1Norm as
        // getAnomalyScore
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        IVisitorFactory<DiVector> visitorFactory = new VisitorFactory<>(
                (tree, y) -> new AnomalyAttributionVisitor(tree.projectToTree(y), tree.getMass()),
                (tree, x) -> x.lift(tree::liftFromTree));
        BinaryOperator<DiVector> accumulator = DiVector::addToLeft;
        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / numberOfTrees);

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Sequential version of attribution corresponding to getAnomalyScoreSequential;
     * The high-low sum in the result should be the same as the scalar score
     * computed by {@link #getAnomalyScore(double[])}.
     *
     * @param point The point being scored.
     * @return anomaly attribution for the given point.
     */
    public DiVector getApproximateAnomalyAttribution(double[] point) {
        return getApproximateAnomalyAttribution(toFloatArray(point));
    }

    public DiVector getApproximateAnomalyAttribution(float[] point) {
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        IVisitorFactory<DiVector> visitorFactory = new VisitorFactory<>(
                (tree, y) -> new AnomalyAttributionVisitor(tree.projectToTree(y), tree.getMass()),
                (tree, x) -> x.lift(tree::liftFromTree));

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions,
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / accumulator.getValuesAccepted());

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Compute a density estimate at the given point.
     * <p>
     * See {@link SimpleInterpolationVisitor} and {@link DensityOutput} for more
     * details about the density computation.
     *
     * @param point The point where the density estimate is made.
     * @return A density estimate.
     */
    @Deprecated
    public DensityOutput getSimpleDensity(double[] point) {
        return getSimpleDensity(toFloatArray(point));
    }

    public DensityOutput getSimpleDensity(float[] point) {

        // density estimation should use sufficiently larger number of samples
        // and only return answers when full

        if (!samplersFull()) {
            return new DensityOutput(dimensions, sampleSize);
        }

        IVisitorFactory<InterpolationMeasure> visitorFactory = new VisitorFactory<>((tree,
                y) -> new SimpleInterpolationVisitor(tree.projectToTree(y), sampleSize, 1.0, centerOfMassEnabled),
                (tree, x) -> x.lift(tree::liftFromTree));
        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions,
                sampleSize, numberOfTrees);

        return new DensityOutput(traverseForest(transformToShingledPoint(point), visitorFactory, collector));
    }

    /**
     * Given a point with missing values, return a new point with the missing values
     * imputed. Each tree in the forest individual produces an imputed value. For
     * 1-dimensional points, the median imputed value is returned. For points with
     * more than 1 dimension, the imputed point with the 25th percentile anomaly
     * score is returned.
     *
     * The first function exposes the distribution.
     *
     * @param point                 A point with missing values.
     * @param numberOfMissingValues The number of missing values in the point.
     * @param missingIndexes        An array containing the indexes of the missing
     *                              values in the point. The length of the array
     *                              should be greater than or equal to the number of
     *                              missing values.
     * @param centrality            a parameter that provides a central estimation
     *                              versus a more random estimation
     * @return A point with the missing values imputed.
     */
    public List<double[]> getConditionalField(double[] point, int numberOfMissingValues, int[] missingIndexes,
            double centrality) {
        return getConditionalField(toFloatArray(point), numberOfMissingValues, missingIndexes, centrality);
    }

    public List<double[]> getConditionalField(float[] point, int numberOfMissingValues, int[] missingIndexes,
            double centrality) {
        checkArgument(numberOfMissingValues > 0, "numberOfMissingValues must be greater than 0");
        checkNotNull(missingIndexes, "missingIndexes must not be null");
        checkArgument(numberOfMissingValues <= missingIndexes.length,
                "numberOfMissingValues must be less than or equal to missingIndexes.length");
        checkArgument(centrality >= 0 && centrality <= 1, "centrality needs to be in range [0,1]");

        if (!isOutputReady()) {
            return new ArrayList<>();
        }

        int[] liftedIndices = transformIndices(missingIndexes, point.length);
        IMultiVisitorFactory<double[]> visitorFactory = (tree, y) -> new ImputeVisitor(y, tree.projectToTree(y),
                liftedIndices, tree.projectMissingIndices(liftedIndices), 1.0);

        Collector<double[], ArrayList<double[]>, ArrayList<double[]>> collector = Collector.of(ArrayList::new,
                ArrayList::add, (left, right) -> {
                    left.addAll(right);
                    return left;
                }, list -> list);

        return traverseForestMulti(transformToShingledPoint(point), visitorFactory, collector);
    }

    /**
     * Given a point with missing values, return a new point with the missing values
     * imputed. Each tree in the forest individual produces an imputed value. For
     * 1-dimensional points, the median imputed value is returned. For points with
     * more than 1 dimension, the imputed point with the 25th percentile anomaly
     * score is returned.
     *
     * @param point                 A point with missing values.
     * @param numberOfMissingValues The number of missing values in the point.
     * @param missingIndexes        An array containing the indexes of the missing
     *                              values in the point. The length of the array
     *                              should be greater than or equal to the number of
     *                              missing values.
     * @return A point with the missing values imputed.
     */
    public double[] imputeMissingValues(double[] point, int numberOfMissingValues, int[] missingIndexes) {
        return imputeMissingValues(toFloatArray(point), numberOfMissingValues, missingIndexes);
    }

    public double[] imputeMissingValues(float[] point, int numberOfMissingValues, int[] missingIndexes) {
        checkArgument(numberOfMissingValues >= 0, "numberOfMissingValues must be greater or equal than 0");
        checkNotNull(missingIndexes, "missingIndexes must not be null");
        checkArgument(numberOfMissingValues <= missingIndexes.length,
                "numberOfMissingValues must be less than or equal to missingIndexes.length");
        checkArgument(point != null, " cannot be null");

        if (!isOutputReady()) {
            return new double[dimensions];
        }
        // checks will be performed in the function call
        List<double[]> conditionalField = getConditionalField(point, numberOfMissingValues, missingIndexes, 1.0);

        if (numberOfMissingValues == 1) {
            // when there is 1 missing value, we sort all the imputed values and return the
            // median
            double[] returnPoint = toDoubleArray(point);
            double[] basicList = conditionalField.stream()
                    .mapToDouble(array -> array[transformIndices(missingIndexes, point.length)[0]]).sorted().toArray();
            returnPoint[missingIndexes[0]] = basicList[numberOfTrees / 2];
            return returnPoint;
        } else {
            // when there is more than 1 missing value, we sort the imputed points by
            // anomaly score and return the point with the 25th percentile anomaly score
            conditionalField.sort(Comparator.comparing(this::getAnomalyScore));
            return conditionalField.get(numberOfTrees / 4);
        }
    }

    /**
     * Given an initial shingled point, extrapolate the stream into the future to
     * produce a forecast. This method is intended to be called when the input data
     * is being shingled, and it works by imputing forward one shingle block at a
     * time.
     *
     * @param point        The starting point for extrapolation.
     * @param horizon      The number of blocks to forecast.
     * @param blockSize    The number of entries in a block. This should be the same
     *                     as the size of a single input to the shingle.
     * @param cyclic       If true then the shingling is cyclic, otherwise it's a
     *                     sliding shingle.
     * @param shingleIndex If cyclic is true, then this should be the current index
     *                     in the shingle. That is, the index where the next point
     *                     added to the shingle would be written. If cyclic is false
     *                     then this value is not used.
     * @return a forecasted time series.
     */
    public double[] extrapolateBasic(double[] point, int horizon, int blockSize, boolean cyclic, int shingleIndex) {
        checkArgument(0 < blockSize && blockSize < dimensions,
                "blockSize must be between 0 and dimensions (exclusive)");
        checkArgument(dimensions % blockSize == 0, "dimensions must be evenly divisible by blockSize");
        checkArgument(0 <= shingleIndex && shingleIndex < dimensions / blockSize,
                "shingleIndex must be between 0 (inclusive) and dimensions / blockSize");

        double[] result = new double[blockSize * horizon];
        int[] missingIndexes = new int[blockSize];
        double[] queryPoint = Arrays.copyOf(point, dimensions);

        if (cyclic) {
            extrapolateBasicCyclic(result, horizon, blockSize, shingleIndex, queryPoint, missingIndexes);
        } else {
            extrapolateBasicSliding(result, horizon, blockSize, queryPoint, missingIndexes);
        }

        return result;
    }

    /**
     * Given an initial shingled point, extrapolate the stream into the future to
     * produce a forecast. This method is intended to be called when the input data
     * is being shingled, and it works by imputing forward one shingle block at a
     * time. If the shingle is cyclic, then this method uses 0 as the shingle index.
     *
     * @param point     The starting point for extrapolation.
     * @param horizon   The number of blocks to forecast.
     * @param blockSize The number of entries in a block. This should be the same as
     *                  the size of a single input to the shingle.
     * @param cyclic    If true then the shingling is cyclic, otherwise it's a
     *                  sliding shingle.
     * @return a forecasted time series.
     */
    public double[] extrapolateBasic(double[] point, int horizon, int blockSize, boolean cyclic) {
        return extrapolateBasic(point, horizon, blockSize, cyclic, 0);
    }

    /**
     * Given a shingle builder, extrapolate the stream into the future to produce a
     * forecast. This method assumes you are passing in the shingle builder used to
     * preprocess points before adding them to this forest.
     *
     * @param builder The shingle builder used to process points before adding them
     *                to the forest.
     * @param horizon The number of blocks to forecast.
     * @return a forecasted time series.
     */
    public double[] extrapolateBasic(ShingleBuilder builder, int horizon) {
        return extrapolateBasic(builder.getShingle(), horizon, builder.getInputPointSize(), builder.isCyclic(),
                builder.getShingleIndex());
    }

    void extrapolateBasicSliding(double[] result, int horizon, int blockSize, double[] queryPoint,
            int[] missingIndexes) {
        int resultIndex = 0;

        Arrays.fill(missingIndexes, 0);
        for (int y = 0; y < blockSize; y++) {
            missingIndexes[y] = dimensions - blockSize + y;
        }

        for (int k = 0; k < horizon; k++) {
            // shift all entries in the query point left by 1 block
            System.arraycopy(queryPoint, blockSize, queryPoint, 0, dimensions - blockSize);

            double[] imputedPoint = imputeMissingValues(queryPoint, blockSize, missingIndexes);
            for (int y = 0; y < blockSize; y++) {
                result[resultIndex++] = queryPoint[dimensions - blockSize + y] = imputedPoint[dimensions - blockSize
                        + y];
            }
        }
    }

    void extrapolateBasicCyclic(double[] result, int horizon, int blockSize, int shingleIndex, double[] queryPoint,
            int[] missingIndexes) {

        int resultIndex = 0;
        int currentPosition = shingleIndex;
        Arrays.fill(missingIndexes, 0);

        for (int k = 0; k < horizon; k++) {
            for (int y = 0; y < blockSize; y++) {
                missingIndexes[y] = (currentPosition + y) % dimensions;
            }

            double[] imputedPoint = imputeMissingValues(queryPoint, blockSize, missingIndexes);

            for (int y = 0; y < blockSize; y++) {
                result[resultIndex++] = queryPoint[(currentPosition + y)
                        % dimensions] = imputedPoint[(currentPosition + y) % dimensions];
            }

            currentPosition = (currentPosition + blockSize) % dimensions;
        }
    }

    /**
     * Extrapolate the stream into the future to produce a forecast. This method is
     * intended to be called when the input data is being shingled internally, and
     * it works by imputing forward one shingle block at a time.
     *
     * @param horizon The number of blocks to forecast.
     * @return a forecasted time series.
     */
    public double[] extrapolate(int horizon) {
        checkArgument(internalShinglingEnabled, "incorrect use");
        IPointStore<?> store = stateCoordinator.getStore();
        return extrapolateBasic(toDoubleArray(lastShingledPoint()), horizon, inputDimensions,
                store.isInternalRotationEnabled(), ((int) nextSequenceIndex()) % shingleSize);
    }

    /**
     * For each tree in the forest, follow the tree traversal path and return the
     * leaf node if the standard Euclidean distance between the query point and the
     * leaf point is smaller than the given threshold. Note that this will not
     * necessarily be the nearest point in the tree, because the traversal path is
     * determined by the random cuts in the tree. If the same leaf point is found in
     * multiple trees, those results will be combined into a single Neighbor in the
     * result.
     *
     * If sequence indexes are disabled for this forest, then the list of sequence
     * indexes will be empty in returned Neighbors.
     *
     * @param point             A point whose neighbors we want to find.
     * @param distanceThreshold The maximum Euclidean distance for a point to be
     *                          considered a neighbor.
     * @return a list of Neighbors, ordered from closest to furthest.
     */
    public List<Neighbor> getNearNeighborsInSample(double[] point, double distanceThreshold) {
        return getNearNeighborsInSample(toFloatArray(point), distanceThreshold);
    }

    public List<Neighbor> getNearNeighborsInSample(float[] point, double distanceThreshold) {
        checkNotNull(point, "point must not be null");
        checkArgument(distanceThreshold > 0, "distanceThreshold must be greater than 0");

        if (!isOutputReady()) {
            return Collections.emptyList();
        }

        IVisitorFactory<Optional<Neighbor>> visitorFactory = (tree, x) -> new NearNeighborVisitor(x, distanceThreshold);

        return traverseForest(transformToShingledPoint(point), visitorFactory, Neighbor.collector());
    }

    /**
     * For each tree in the forest, follow the tree traversal path and return the
     * leaf node. Note that this will not necessarily be the nearest point in the
     * tree, because the traversal path is determined by the random cuts in the
     * tree. If the same leaf point is found in multiple trees, those results will
     * be combined into a single Neighbor in the result.
     *
     * If sequence indexes are disabled for this forest, then sequenceIndexes will
     * be empty in the returned Neighbors.
     *
     * @param point A point whose neighbors we want to find.
     * @return a list of Neighbors, ordered from closest to furthest.
     */
    public List<Neighbor> getNearNeighborsInSample(double[] point) {
        return getNearNeighborsInSample(point, Double.POSITIVE_INFINITY);
    }

    /**
     * @return true if all samplers are ready to output results.
     */
    public boolean isOutputReady() {
        return outputReady || (outputReady = components.stream().allMatch(IComponentModel::isOutputReady));
    }

    /**
     * @return true if all samplers in the forest are full.
     */
    public boolean samplersFull() {
        return stateCoordinator.getTotalUpdates() >= sampleSize;
    }

    /**
     * Returns the total number updates to the forest.
     *
     * The count of updates is represented with long type and may overflow.
     *
     * @return the total number of updates to the forest.
     */
    public long getTotalUpdates() {
        return stateCoordinator.getTotalUpdates();
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        private int dimensions;
        private int sampleSize = DEFAULT_SAMPLE_SIZE;
        private Optional<Integer> outputAfter = Optional.empty();
        private int numberOfTrees = DEFAULT_NUMBER_OF_TREES;
        private Optional<Double> timeDecay = Optional.empty();
        private Optional<Long> randomSeed = Optional.empty();
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        private boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        private boolean parallelExecutionEnabled = DEFAULT_PARALLEL_EXECUTION_ENABLED;
        private Optional<Integer> threadPoolSize = Optional.empty();
        private boolean directLocationMapEnabled = DEFAULT_DIRECT_LOCATION_MAP;
        private double boundingBoxCacheFraction = DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        private int shingleSize = DEFAULT_SHINGLE_SIZE;
        protected boolean dynamicResizingEnabled = DEFAULT_DYNAMIC_RESIZING_ENABLED;
        private boolean internalShinglingEnabled = DEFAULT_INTERNAL_SHINGLING_ENABLED;
        protected boolean internalRotationEnabled = DEFAULT_INTERNAL_ROTATION_ENABLED;
        protected Optional<Integer> initialPointStoreSize = Optional.empty();
        protected double initialAcceptFraction = DEFAULT_INITIAL_ACCEPT_FRACTION;

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

        public T initialPointStoreSize(int initialPointStoreSize) {
            this.initialPointStoreSize = Optional.of(initialPointStoreSize);
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
            this.internalShinglingEnabled = internalShinglingEnabled;
            return (T) this;
        }

        public T internalRotationEnabled(boolean internalRotationEnabled) {
            this.internalRotationEnabled = internalRotationEnabled;
            return (T) this;
        }

        public T dynamicResizingEnabled(boolean dynamicResizingEnabled) {
            this.dynamicResizingEnabled = dynamicResizingEnabled;
            return (T) this;
        }

        @Deprecated
        public T precision(Precision precision) {
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

        public RandomCutForest build() {
            return new RandomCutForest(this);
        }

        public Random getRandom() {
            // If a random seed was given, use it to create a new Random. Otherwise, call
            // the 0-argument constructor
            return randomSeed.map(Random::new).orElseGet(Random::new);
        }
    }

    /**
     * Score a point using the given scoring functions.
     *
     * @param point                   input point being scored
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    the function that applies if input is equal to
     *                                a previously seen sample in a leaf
     * @param unseen                  if the input does not have a match in the
     *                                leaves
     * @param damp                    damping function based on the duplicity of the
     *                                previously seen samples
     * @return anomaly score
     */
    public double getDynamicScore(float[] point, int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp) {

        checkArgument(ignoreLeafMassThreshold >= 0, "ignoreLeafMassThreshold should be greater than or equal to 0");

        if (!isOutputReady()) {
            return 0.0;
        }

        VisitorFactory<Double> visitorFactory = new VisitorFactory<>((tree, y) -> new DynamicScoreVisitor(
                tree.projectToTree(y), tree.getMass(), ignoreLeafMassThreshold, seen, unseen, damp));
        BinaryOperator<Double> accumulator = Double::sum;

        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Similar to above but now the scoring takes in a function of Bounding Box to
     * probabilities (vector over the dimensions); and produces a score af-if the
     * tree were built using that function (when in reality the tree is an RCF).
     * Changing the defaultRCFgVec function to some other function f() will provide
     * a mechanism of dynamic scoring for trees that are built using f() which is
     * the purpose of TransductiveScalarScore visitor. Note that the answer is an
     * MCMC simulation and is not normalized (because the scoring functions are
     * flexible and unknown) and over a small number of trees the errors can be
     * large specially if vecSep is very far from defaultRCFgVec
     *
     * Given the large number of possible sources of distortion, ignoreLeafThreshold
     * is not supported.
     *
     * @param point  point to be scored
     * @param seen   the score function for seen point
     * @param unseen score function for unseen points
     * @param damp   dampening the score for duplicates
     * @param vecSep the function of (BoundingBox) -&gt; array of probabilities
     * @return the simuated score
     */

    public double getDynamicSimulatedScore(float[] point, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp,
            Function<IBoundingBoxView, double[]> vecSep) {

        if (!isOutputReady()) {
            return 0.0;
        }

        VisitorFactory<Double> visitorFactory = new VisitorFactory<>(
                (tree, y) -> new SimulatedTransductiveScalarScoreVisitor(tree.projectToTree(y), tree.getMass(), seen,
                        unseen, damp, CommonUtils::defaultRCFgVecFunction, vecSep));
        BinaryOperator<Double> accumulator = Double::sum;

        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Score a point using the given scoring functions. This method will
     * short-circuit before visiting all trees if the scores that are returned from
     * a subset of trees appears to be converging to a given value. See
     * {@link OneSidedConvergingDoubleAccumulator} for more about convergence.
     *
     * @param point                   input point
     * @param precision               controls early convergence
     * @param highIsCritical          this is true for the default scoring function.
     *                                If the user wishes to use a different scoring
     *                                function where anomaly scores are low values
     *                                (for example, height in tree) then this should
     *                                be set to false.
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    scoring function when the input matches some
     *                                tuple in the leaves
     * @param unseen                  scoring function when the input is not found
     * @param damp                    dampening function for duplicates which are
     *                                same as input (applies with seen)
     * @return the dynamic score under sequential early stopping
     */
    public double getApproximateDynamicScore(float[] point, double precision, boolean highIsCritical,
            int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp) {

        checkArgument(ignoreLeafMassThreshold >= 0, "ignoreLeafMassThreshold should be greater than or equal to 0");

        if (!isOutputReady()) {
            return 0.0;
        }

        VisitorFactory<Double> visitorFactory = new VisitorFactory<>((tree, y) -> new DynamicScoreVisitor(
                tree.projectToTree(y), tree.getMass(), ignoreLeafMassThreshold, seen, unseen, damp));

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(highIsCritical, precision,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Same as above, but for dynamic scoring. See the params of
     * getDynamicScoreParallel
     *
     * @param point                   point to be scored
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    score function for seen points
     * @param unseen                  score function for unseen points
     * @param newDamp                 dampening function for duplicates in the seen
     *                                function
     * @return dynamic scoring attribution DiVector
     */
    public DiVector getDynamicAttribution(float[] point, int ignoreLeafMassThreshold,
            BiFunction<Double, Double, Double> seen, BiFunction<Double, Double, Double> unseen,
            BiFunction<Double, Double, Double> newDamp) {

        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        VisitorFactory<DiVector> visitorFactory = new VisitorFactory<>(
                (tree, y) -> new DynamicAttributionVisitor(tree.projectToTree(y), tree.getMass(),
                        ignoreLeafMassThreshold, seen, unseen, newDamp),
                (tree, x) -> x.lift(tree::liftFromTree));
        BinaryOperator<DiVector> accumulator = DiVector::addToLeft;
        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / numberOfTrees);

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

    /**
     * Atrribution for dynamic sequential scoring; getL1Norm() should agree with
     * getDynamicScoringSequential
     *
     * @param point                   input
     * @param precision               parameter to stop early stopping
     * @param highIsCritical          are high values anomalous (otherwise low
     *                                values are anomalous)
     * @param ignoreLeafMassThreshold we ignore leaves with mass equal/below *
     *                                threshold
     * @param seen                    function for scoring points that have been
     *                                seen before
     * @param unseen                  function for scoring points not seen in tree
     * @param newDamp                 dampening function based on duplicates
     * @return attribution DiVector of the score
     */
    public DiVector getApproximateDynamicAttribution(float[] point, double precision, boolean highIsCritical,
            int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> newDamp) {

        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        VisitorFactory<DiVector> visitorFactory = new VisitorFactory<>((tree, y) -> new DynamicAttributionVisitor(y,
                tree.getMass(), ignoreLeafMassThreshold, seen, unseen, newDamp),
                (tree, x) -> x.lift(tree::liftFromTree));

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions,
                highIsCritical, precision, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<DiVector, DiVector> finisher = vector -> vector.scale(1.0 / accumulator.getValuesAccepted());

        return traverseForest(transformToShingledPoint(point), visitorFactory, accumulator, finisher);
    }

}
