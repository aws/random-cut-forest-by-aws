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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import com.amazon.randomcutforest.anomalydetection.AnomalyAttributionVisitor;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.executor.AbstractForestUpdateExecutor;
import com.amazon.randomcutforest.executor.IUpdateCoordinator;
import com.amazon.randomcutforest.executor.ParallelForestTraversalExecutor;
import com.amazon.randomcutforest.executor.ParallelForestUpdateExecutor;
import com.amazon.randomcutforest.executor.PassThroughCoordinator;
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
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;
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
     * If the user doesn't specify an explicit lambda value, then we set it to the
     * inverse of this coefficient times sample size.
     */
    public static final double DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_LAMBDA = 10.0;

    /**
     * Default number of trees to use in the forest.
     */
    public static final int DEFAULT_NUMBER_OF_TREES = 50;

    /**
     * By default, trees will not store sequence indexes.
     */
    public static final boolean DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED = false;

    /**
     * By default, trees will not create indexed references.
     */
    public static final boolean DEFAULT_COMPACT_ENABLED = false;

    /**
     * Default floating-point precision for internal data structures.
     */
    public static final Precision DEFAULT_PRECISION = Precision.DOUBLE;

    /**
     * By default, bounding boxes will be used. Disabling this will force
     * enableCompact .
     */
    public static final boolean DEFAULT_BOUNDING_BOX_CACHE_ENABLED = true;

    /**
     * By default, nodes will not store center of mass.
     */
    public static final boolean DEFAULT_CENTER_OF_MASS_ENABLED = false;

    /**
     * By default RCF is unaware of shingle size
     */
    public static final int DEFAULT_SHINGLE_SIZE = 1;

    /**
     * internal shingling is not enabled by default
     */
    public static final boolean DEFAULT_INTERNAL_SHINGLING_ENABLED = false;

    /**
     * rotation of shingles (applicable only for internal shingling) not true by
     * default
     */
    public static final boolean DEFAULT_ROTATE_INTERNAL_SHINGLES_ENABLED = false;

    /**
     * pointstore can increase dynamically by default; helps save space by not
     * provisioning the long term worst case sizes
     */
    public static final boolean DEFAULT_DYNAMICALLY_RESIZE_POINTSTORE_ENABLED = true;

    public static final boolean DEFAULT_EXTERNAL_POINTSTORE_USED = false;

    /**
     * Parallel execution is not enabled by default.
     */
    public static final boolean DEFAULT_PARALLEL_EXECUTION_ENABLED = false;

    public static final boolean DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL = true;

    public static final double DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION = 0.1;

    public static final int DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED = 5;

    /**
     * Random number generator used by the forest.
     */
    protected Random rng;
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
     * dimensionality of non-shingled data
     */
    protected final int baseDimensions;
    /**
     * The number of points required by stream samplers before results are returned.
     */
    protected final int outputAfter;
    /**
     * The number of trees in this forest.
     */
    protected final int numberOfTrees;
    /**
     * The decay factor (lambda value) used by stream samplers in this forest.
     */
    protected double lambda;
    /**
     * Store the time information
     */
    protected final boolean storeSequenceIndexesEnabled;
    /**
     * Enable compact representation
     */
    protected final boolean compactEnabled;
    /**
     * The preferred floating point precision to use in internal data structures.
     * This will affect runtime memory and the size of a serialized model.
     */
    protected final Precision precision;
    /**
     * The following set to false saves space, but enforces compact representation.
     */
    protected final boolean boundingBoxCachingEnabled;
    /**
     * Enable center of mass at internal nodes
     */
    protected final boolean centerOfMassEnabled;

    /**
     * enable internal shingling
     */
    protected final boolean internalShinglingEnabled;

    /**
     * consider rotation of internal shingles
     */
    protected final boolean rotateInternalShinglesEnabled;

    /**
     * consider dynamically resizing pointstore
     */
    protected final boolean dynamicallyResizePointStoreEnabled;

    /**
     * use of external point store
     */
    protected final boolean externalPointStoreUsed;

    /**
     * Enable parallel execution.
     */
    protected final boolean parallelExecutionEnabled;
    /**
     * Number of threads to use in the thread pool if parallel execution is enabled.
     */
    protected final int threadPoolSize;

    protected IUpdateCoordinator<?> updateCoordinator;
    protected ComponentList<?> components;

    /**
     * An implementation of forest traversal algorithms.
     */
    protected AbstractForestTraversalExecutor traversalExecutor;

    /**
     * An implementation of forest update algorithms.
     */
    protected AbstractForestUpdateExecutor<?> updateExecutor;

    public <Q> RandomCutForest(Builder<?> builder, IUpdateCoordinator<Q> updateCoordinator, ComponentList<Q> components,
            Random rng) {
        this(builder, false);

        checkNotNull(updateCoordinator, "updateCoordinator must not be null");
        checkNotNull(components, "componentModels must not be null");
        checkNotNull(rng, "rng must not be null");

        this.updateCoordinator = updateCoordinator;
        this.components = components;
        this.rng = rng;
        initExecutors(updateCoordinator, components);
        if (builder.compactEnabled) {
            updateExecutor.forEachTree(t -> t.usePointStore(((PointStoreCoordinator) updateCoordinator).getStore()));
        }
    }

    public RandomCutForest(Builder<?> builder) {
        this(builder, false);
        rng = builder.getRandom();

        int maxCapacity = 1 + numberOfTrees * sampleSize;
        int storeSize = (builder.initialPointStoreSize.isPresent()) ? builder.initialPointStoreSize.get() : maxCapacity;
        if (dynamicallyResizePointStoreEnabled) {
            storeSize = (builder.initialPointStoreSize.isPresent()) ? builder.initialPointStoreSize.get()
                    : 2 * sampleSize;
        }

        if (precision == Precision.SINGLE) {
            initCompactFloat(new PointStoreFloat(dimensions, shingleSize, maxCapacity, storeSize,
                    internalShinglingEnabled, true, (shingleSize == 1), rotateInternalShinglesEnabled));
        } else if (compactEnabled) {
            initCompactDouble(new PointStoreDouble(dimensions, shingleSize, maxCapacity, storeSize,
                    internalShinglingEnabled, true, (shingleSize == 1), rotateInternalShinglesEnabled));
        } else {
            initNonCompact();
        }
    }

    private void initCompactDouble(IPointStore<double[]> pointStore) {

        IUpdateCoordinator<Integer> updateCoordinator = new PointStoreCoordinator(pointStore);
        ComponentList<Integer> components = new ComponentList<>(numberOfTrees);
        for (int i = 0; i < numberOfTrees; i++) {
            ITree<Integer> tree = new CompactRandomCutTreeDouble(sampleSize, rng.nextLong(), pointStore,
                    boundingBoxCachingEnabled, centerOfMassEnabled, storeSequenceIndexesEnabled);
            IStreamSampler<Integer> sampler = new CompactSampler(sampleSize, lambda, rng.nextLong(),
                    storeSequenceIndexesEnabled);
            components.add(new SamplerPlusTree<>(sampler, tree));
        }
        this.updateCoordinator = updateCoordinator;
        this.components = components;
        initExecutors(updateCoordinator, components);
    }

    private void initCompactFloat(IPointStore<float[]> pointStore) {
        IUpdateCoordinator<Integer> updateCoordinator = new PointStoreCoordinator(pointStore);
        ComponentList<Integer> components = new ComponentList<>(numberOfTrees);
        for (int i = 0; i < numberOfTrees; i++) {
            ITree<Integer> tree = new CompactRandomCutTreeFloat(sampleSize, rng.nextLong(), pointStore,
                    boundingBoxCachingEnabled, centerOfMassEnabled, storeSequenceIndexesEnabled);
            IStreamSampler<Integer> sampler = new CompactSampler(sampleSize, lambda, rng.nextLong(),
                    storeSequenceIndexesEnabled);
            components.add(new SamplerPlusTree<>(sampler, tree));
        }
        this.updateCoordinator = updateCoordinator;
        this.components = components;
        initExecutors(updateCoordinator, components);
    }

    private void initNonCompact() {
        IUpdateCoordinator<double[]> updateCoordinator = new PassThroughCoordinator();
        ComponentList<double[]> components = new ComponentList<>(numberOfTrees);
        for (int i = 0; i < numberOfTrees; i++) {
            ITree<double[]> tree = new RandomCutTree(rng.nextLong(), boundingBoxCachingEnabled, centerOfMassEnabled,
                    storeSequenceIndexesEnabled);
            IStreamSampler<double[]> sampler = new SimpleStreamSampler<>(sampleSize, lambda, rng.nextLong(),
                    storeSequenceIndexesEnabled);
            components.add(new SamplerPlusTree<>(sampler, tree));
        }
        this.updateCoordinator = updateCoordinator;
        this.components = components;
        initExecutors(updateCoordinator, components);
    }

    private <Q> void initExecutors(IUpdateCoordinator<Q> updateCoordinator, ComponentList<Q> components) {
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
    private RandomCutForest(Builder<?> builder, boolean notUsed) {
        checkArgument(builder.numberOfTrees > 0, "numberOfTrees must be greater than 0");
        checkArgument(builder.sampleSize > 0, "sampleSize must be greater than 0");
        builder.outputAfter.ifPresent(n -> {
            checkArgument(n > 0, "outputAfter must be greater than 0");
            checkArgument(n <= builder.sampleSize, "outputAfter must be smaller or equal to sampleSize");
        });
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        builder.lambda.ifPresent(lambda -> {
            checkArgument(lambda >= 0, "lambda must be greater than or equal to 0");
        });
        builder.threadPoolSize.ifPresent(n -> checkArgument((n > 0) || ((n == 0) && !builder.parallelExecutionEnabled),
                "threadPoolSize must be greater/equal than 0. To disable thread pool, set parallel execution to 'false'."));
        checkArgument(builder.precision == Precision.DOUBLE || builder.compactEnabled,
                "single precision is only supported for compact trees");
        checkArgument(builder.internalShinglingEnabled || builder.shingleSize == 1
                || builder.dimensions % builder.shingleSize == 0, "wrong shingle size");
        checkArgument(!builder.rotateInternalShinglesEnabled || builder.internalShinglingEnabled,
                "enable internal shingling to enable internal rotation");

        numberOfTrees = builder.numberOfTrees;
        sampleSize = builder.sampleSize;
        outputAfter = builder.outputAfter.orElse((int) (sampleSize * DEFAULT_OUTPUT_AFTER_FRACTION));
        internalShinglingEnabled = builder.internalShinglingEnabled;
        shingleSize = builder.shingleSize;
        dimensions = internalShinglingEnabled ? builder.dimensions * shingleSize : builder.dimensions;
        baseDimensions = internalShinglingEnabled ? dimensions / shingleSize : dimensions;
        lambda = builder.lambda.orElse(1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_LAMBDA * sampleSize));
        storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        centerOfMassEnabled = builder.centerOfMassEnabled;
        parallelExecutionEnabled = builder.parallelExecutionEnabled;
        compactEnabled = builder.compactEnabled;
        precision = builder.precision;
        boundingBoxCachingEnabled = builder.boundingBoxCachingEnabled;
        dynamicallyResizePointStoreEnabled = builder.dynamicallyResizePointStoreEnabled;
        rotateInternalShinglesEnabled = builder.rotateInternalShinglesEnabled;
        externalPointStoreUsed = builder.externalPointStoreUsed;

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
     * @return return the decay factor (lambda value) used by stream samplers in
     *         this forest.
     */
    public double getLambda() {
        return lambda;
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
        return precision;
    }

    /**
     * @return true if points are saved with sequence indexes, false otherwise.
     */
    public boolean isCompactEnabled() {
        return compactEnabled;
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

    public boolean isBoundingBoxCachingEnabled() {
        return boundingBoxCachingEnabled;
    }

    public boolean isDynamicallyResizePointstoreEnabled() {
        return dynamicallyResizePointStoreEnabled;
    }

    public boolean isInternalShinglingEnabled() {
        return internalShinglingEnabled;
    }

    public boolean isRotateInternalShinglesEnabled() {
        return rotateInternalShinglesEnabled;
    }

    public boolean getExternalPointStoreUsed() {
        return externalPointStoreUsed;
    }

    /**
     * @return the number of threads in the thread pool if parallel execution is
     *         enabled, 0 otherwise.
     */
    public int getThreadPoolSize() {
        return threadPoolSize;
    }

    public IUpdateCoordinator<?> getUpdateCoordinator() {
        return updateCoordinator;
    }

    public ComponentList<?> getComponents() {
        return components;
    }

    public boolean isExternalPointStoreUsed() {
        return externalPointStoreUsed;
    }

    /**
     * Update the forest with the given point. The point is submitted to each
     * sampler in the forest. If the sampler accepts the point, the point is
     * submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        checkNotNull(point, "point must not be null");
        checkArgument(internalShinglingEnabled || point.length == dimensions,
                String.format("point.length must equal %d", dimensions));
        checkArgument(!internalShinglingEnabled || point.length == baseDimensions,
                String.format("point.length must equal %d for internal shingling", baseDimensions));

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
        checkArgument(0 <= cacheFraction && cacheFraction <= 1, String.format("fraction must be in [0,1]"));
        updateExecutor.forEachTree(t -> t.setBoundingBoxCacheFraction(cacheFraction));
    }

    /**
     * changes the setting of time dependent sampling on the fly
     * 
     * @param lambda new value of sampling rate
     */
    public void setLambda(double lambda) {
        checkArgument(0 <= lambda, String.format("lambda cannot be negative"));
        this.lambda = lambda;
        updateExecutor.forEachSampler(t -> t.setTimeDecay(lambda));
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to
     * {@link RandomCutTree#traverse(double[], Function)}. The results from all the
     * trees are combined using the accumulator and then transformed using the
     * finisher before being returned. Trees are visited in parallel using
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
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

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
     * {@link RandomCutTree#traverse(double[], Function)}. The results from
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
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

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
     * {@link RandomCutTree#traverse(double[], Function)}. The results from all the
     * trees are combined using the {@link ConvergingAccumulator}, and the method
     * stops visiting trees after convergence is reached. The result is transformed
     * using the finisher before being returned.
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
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
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
     * {@link RandomCutTree#traverseMulti(double[], Function)}. The results from all
     * the trees are combined using the accumulator and then transformed using the
     * finisher before being returned.
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
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
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
     * {@link RandomCutTree#traverseMulti(double[], Function)}. The results from
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
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
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
    public double getAnomalyScore(double[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<Double>> visitorFactory = tree -> new AnomalyScoreVisitor(newPoint, tree.getMass());
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(newPoint, visitorFactory, accumulator, finisher);
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

    public double getApproximateAnomalyScore(double[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<Double>> visitorFactory = tree -> new AnomalyScoreVisitor(newPoint, tree.getMass());

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        return traverseForest(newPoint, visitorFactory, accumulator, finisher);
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
        // this will return the same (modulo floating point summation) L1Norm as
        // getAnomalyScore
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }
        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<DiVector>> visitorFactory = tree -> new AnomalyAttributionVisitor(newPoint,
                tree.getMass());
        BinaryOperator<DiVector> accumulator = DiVector::addToLeft;
        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / numberOfTrees);

        return traverseForest(newPoint, visitorFactory, accumulator, finisher);
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
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<DiVector>> visitorFactory = tree -> new AnomalyAttributionVisitor(newPoint,
                tree.getMass());

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions,
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<DiVector, DiVector> finisher = vector -> vector.scale(1.0 / accumulator.getValuesAccepted());

        return traverseForest(newPoint, visitorFactory, accumulator, finisher);
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
    public DensityOutput getSimpleDensity(double[] point) {

        // density estimation should use sufficiently larger number of samples
        // and only return answers when full

        if (!samplersFull()) {
            return new DensityOutput(dimensions, sampleSize);
        }

        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<InterpolationMeasure>> visitorFactory = tree -> new SimpleInterpolationVisitor(
                newPoint, sampleSize, 1.0, centerOfMassEnabled); // self
        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions,
                sampleSize, numberOfTrees);

        return new DensityOutput(traverseForest(newPoint, visitorFactory, collector));
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
     * @return A point with the missing values imputed.
     */
    public ArrayList<double[]> getCentralConditionalField(double[] point, int numberOfMissingValues,
            int[] missingIndexes) {
        checkArgument(numberOfMissingValues > 0, "numberOfMissingValues must be greater than or equal to 0");

        // We check this condition in traverseForest, but we need to check it here as
        // well in case we need to copy the
        // point in the next block
        checkNotNull(point, "point must not be null");

        checkNotNull(missingIndexes, "missingIndexes must not be null");
        checkArgument(numberOfMissingValues <= missingIndexes.length,
                "numberOfMissingValues must be less than or equal to missingIndexes.length");

        if (!isOutputReady()) {
            return new ArrayList<>();
        }

        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        int[] indexList = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformIndices(missingIndexes)
                : missingIndexes;
        Function<ITree<?>, MultiVisitor<double[]>> visitorFactory = tree -> new ImputeVisitor(newPoint,
                numberOfMissingValues, indexList);

        Collector<double[], ArrayList<double[]>, ArrayList<double[]>> collector = Collector.of(ArrayList::new,
                ArrayList::add, (left, right) -> {
                    left.addAll(right);
                    return left;
                }, list -> list);

        return traverseForestMulti(point, visitorFactory, collector);

    }

    public double[] imputeMissingValues(double[] point, int numberOfMissingValues, int[] missingIndexes) {
        checkArgument(numberOfMissingValues >= 0, "numberOfMissingValues must be greater than or equal to 0");

        // We check this condition in traverseForest, but we need to check it here as
        // well in case we need to copy the
        // point in the next block
        checkNotNull(point, "point must not be null");

        if (numberOfMissingValues == 0) {
            return Arrays.copyOf(point, point.length);
        }

        checkNotNull(missingIndexes, "missingIndexes must not be null");
        checkArgument(numberOfMissingValues <= missingIndexes.length,
                "numberOfMissingValues must be less than or equal to missingIndexes.length");

        if (!isOutputReady()) {
            return new double[point.length];
        }
        ArrayList<double[]> conditionalField = getCentralConditionalField(point, numberOfMissingValues, missingIndexes);

        if (numberOfMissingValues == 1) {
            // when there is 1 missing value, we sort all the imputed values and return the
            // median
            double[] returnPoint = Arrays.copyOf(point, point.length);
            double[] basicList = conditionalField.stream().mapToDouble(array -> array[missingIndexes[0]]).sorted()
                    .toArray();
            returnPoint[missingIndexes[0]] = basicList[numberOfTrees / 2];
            return returnPoint;
        } else {
            // when there is more than 1 missing value, we sort the imputed points by
            // anomaly score and
            // return the point with the 25th percentile anomaly score
            conditionalField.sort(Comparator.comparing(this::getAnomalyScore));
            return conditionalField.get(numberOfTrees / 4);
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
        IPointStore<?> store = ((PointStoreCoordinator) updateCoordinator).getStore();
        return extrapolateBasic(store.getInternalShingle(), horizon, baseDimensions, rotateInternalShinglesEnabled,
                ((int) store.getLastTimeStamp()) % dimensions);
    }

    public double[] transformToShingledPoint(double[] point) {
        checkArgument(internalShinglingEnabled, "incorrect use");
        return ((PointStoreCoordinator) updateCoordinator).getStore().transformToShingledPoint(point);
    }

    public double[] lastShingledPoint() {
        checkArgument(internalShinglingEnabled, "incorrect use");
        return ((PointStoreCoordinator) updateCoordinator).getStore().getInternalShingle();
    }

    public long timeStamp() {
        checkArgument(internalShinglingEnabled, "incorrect use");
        return ((PointStoreCoordinator) updateCoordinator).getStore().getLastTimeStamp();
    }

    public int[] transformIndices(int[] indexList) {
        return (!internalShinglingEnabled) ? indexList
                : ((PointStoreCoordinator) updateCoordinator).getStore().transformIndices(indexList);
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
        checkNotNull(point, "point must not be null");
        checkArgument(distanceThreshold > 0, "distanceThreshold must be greater than 0");

        if (!isOutputReady()) {
            return Collections.emptyList();
        }
        double[] newPoint = (internalShinglingEnabled && point.length == baseDimensions)
                ? transformToShingledPoint(point)
                : point;
        Function<ITree<?>, Visitor<Optional<Neighbor>>> visitorFactory = tree -> new NearNeighborVisitor(newPoint,
                distanceThreshold);

        return traverseForest(newPoint, visitorFactory, Neighbor.collector());
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
        return updateCoordinator.getTotalUpdates() >= outputAfter;
    }

    /**
     * @return true if all samplers in the forest are full.
     */
    public boolean samplersFull() {
        return updateCoordinator.getTotalUpdates() >= sampleSize;
    }

    /**
     * Returns the total number updates to the forest.
     *
     * The count of updates is represented with long type and may overflow.
     *
     * @return the total number of updates to the forest.
     */
    public long getTotalUpdates() {
        return updateCoordinator.getTotalUpdates();
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        private int dimensions;
        private int sampleSize = DEFAULT_SAMPLE_SIZE;
        private Optional<Integer> outputAfter = Optional.empty();
        private int numberOfTrees = DEFAULT_NUMBER_OF_TREES;
        private Optional<Double> lambda = Optional.empty();
        private Optional<Long> randomSeed = Optional.empty();
        private boolean compactEnabled = DEFAULT_COMPACT_ENABLED;
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        private boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        private boolean parallelExecutionEnabled = DEFAULT_PARALLEL_EXECUTION_ENABLED;
        private Optional<Integer> threadPoolSize = Optional.empty();
        private Precision precision = DEFAULT_PRECISION;
        private boolean boundingBoxCachingEnabled = DEFAULT_BOUNDING_BOX_CACHE_ENABLED;
        private int shingleSize = DEFAULT_SHINGLE_SIZE;
        private boolean internalShinglingEnabled = DEFAULT_INTERNAL_SHINGLING_ENABLED;
        private boolean rotateInternalShinglesEnabled = DEFAULT_ROTATE_INTERNAL_SHINGLES_ENABLED;
        private boolean dynamicallyResizePointStoreEnabled = DEFAULT_DYNAMICALLY_RESIZE_POINTSTORE_ENABLED;
        private Optional<Integer> initialPointStoreSize = Optional.empty();
        private boolean externalPointStoreUsed = DEFAULT_EXTERNAL_POINTSTORE_USED;

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

        public T lambda(double lambda) {
            this.lambda = Optional.of(lambda);
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = Optional.of(randomSeed);
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
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

        public T internalShinglingEnabled(boolean enableInternalShingling) {
            this.internalShinglingEnabled = enableInternalShingling;
            return (T) this;
        }

        public T rotateInternalShinglesEnabled(boolean rotateInternalShingles) {
            this.rotateInternalShinglesEnabled = rotateInternalShingles;
            return (T) this;
        }

        public T dynamicallyResizePointStoreEnabled(boolean dynamicallyResizePointStoreEnabled) {
            this.dynamicallyResizePointStoreEnabled = dynamicallyResizePointStoreEnabled;
            return (T) this;
        }

        public T compactEnabled(boolean compactEnabled) {
            this.compactEnabled = compactEnabled;
            return (T) this;
        }

        public T precision(Precision precision) {
            this.precision = precision;
            return (T) this;
        }

        public T boundingBoxCachingEnabled(boolean boundingBoxCachingEnabled) {
            this.boundingBoxCachingEnabled = boundingBoxCachingEnabled;
            return (T) this;
        }

        public T externalPointStoreUsed(boolean externalPointStoreUsed) {
            this.externalPointStoreUsed = externalPointStoreUsed;
            return (T) this;
        }

        public T initialPointStoreSize(int initialPointStoreSize) {
            this.initialPointStoreSize = Optional.of(initialPointStoreSize);
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
}
