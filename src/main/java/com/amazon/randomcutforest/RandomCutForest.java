/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.ShingleBuilder;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

/**
 * The RandomCutForest class is the interface to the algorithms in this package, and includes methods for anomaly
 * detection, anomaly detection with attribution, density estimation, imputation, and forecasting. A Random Cut Forest
 * is a collection of Random Cut Trees and stream samplers. When an update call is made to a Random Cut Forest, each
 * sampler is independently updated with the submitted (and if the point is accepted by the sampler, then the
 * corresponding Random Cut Tree is also updated. Similarly, when an algorithm method is called, the Random Cut Forest
 * proxies to the trees which implement the actual scoring logic. The Random Cut Forest then combines partial results
 * into a final results.
 */
public class RandomCutForest {

    /**
     * Default sample size. This is the number of points retained by the stream sampler.
     */
    public static final int DEFAULT_SAMPLE_SIZE = 256;

    /**
     * Default fraction used to compute the amount of points required by stream samplers before results are returned.
     */
    public static final double DEFAULT_OUTPUT_AFTER_FRACTION = 0.25;

    /**
     * Default decay value to use in the stream sampler.
     */
    public static final double DEFAULT_LAMBDA = 1e-5;

    /**
     * Default number of trees to use in the forest.
     */
    public static final int DEFAULT_NUMBER_OF_TREES = 100;

    /**
     * By default, trees will not store sequence indexes.
     */
    public static final boolean DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED = false;

    /**
     * By default, nodes will not store center of mass.
     */
    public static final boolean DEFAULT_CENTER_OF_MASS_ENABLED = false;

    /**
     * Parallel execution is enabled by default.
     */
    public static final boolean DEFAULT_PARALLEL_EXECUTION_ENABLED = true;

    public static final boolean DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL = true;

    public static final double DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION = 0.1;

    public static final int DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED = 5;

    /**
     * Random number generator used by the forest.
     */
    protected final Random rng;
    /**
     * The number of dimensions in the input data.
     */
    protected final int dimensions;
    /**
     * The sample size used by stream samplers in this forest.
     */
    protected final int sampleSize;
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
    protected final double lambda;
    /**
     * Store the time information
     */
    protected final boolean storeSequenceIndexesEnabled;
    /**
     * Enable center of mass at internal nodes
     */
    protected final boolean centerOfMassEnabled;
    /**
     * Enable parallel execution.
     */
    protected final boolean parallelExecutionEnabled;
    /**
     * Number of threads to use in the threadpool if parallel execution is enabled.
     */
    protected final int threadPoolSize;
    /**
     * An implementation of forest traversal algorithms.
     */
    protected final AbstractForestTraversalExecutor executor;

    protected RandomCutForest(Builder<?> builder) {
        checkArgument(builder.numberOfTrees > 0, "numberOfTrees must be greater than 0");
        checkArgument(builder.sampleSize > 0, "sampleSize must be greater than 0");
        builder.outputAfter.ifPresent(n -> {
            checkArgument(n > 0, "outputAfter must be greater than 0");
            checkArgument(n <= builder.sampleSize, "outputAfter must be smaller or equal to sampleSize");
        });
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        checkArgument(builder.lambda >= 0, "lambda must be greater than or equal to 0");
        builder.threadPoolSize.ifPresent(n ->
                checkArgument(n > 0, "threadPoolSize must be greater than 0. To disable thread pool, set parallel execution to 'false'."));

        numberOfTrees = builder.numberOfTrees;
        sampleSize = builder.sampleSize;
        outputAfter = builder.outputAfter.orElse((int) (sampleSize * DEFAULT_OUTPUT_AFTER_FRACTION));
        dimensions = builder.dimensions;
        lambda = builder.lambda;
        storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        centerOfMassEnabled = builder.centerOfMassEnabled;
        parallelExecutionEnabled = builder.parallelExecutionEnabled;
        ArrayList<TreeUpdater> treeUpdaters = new ArrayList<>(numberOfTrees);

        // If a random seed was given, use it to create a new Random. Otherwise, call the 0-argument constructor
        rng = builder.randomSeed.map(Random::new).orElseGet(Random::new);

        for (int i = 0; i < numberOfTrees; i++) {
            SimpleStreamSampler sampler = new SimpleStreamSampler(sampleSize, lambda, rng.nextLong());
            RandomCutTree tree = RandomCutTree.builder()
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled)
                    .centerOfMassEnabled(centerOfMassEnabled)
                    .randomSeed(rng.nextLong())
                    .build();
            TreeUpdater updater = new TreeUpdater(sampler, tree);
            treeUpdaters.add(updater);
        }

        if (parallelExecutionEnabled) {
            // If the user specified a thread pool size, use it. Otherwise, use available processors - 1.
            threadPoolSize = builder.threadPoolSize.orElse(Runtime.getRuntime().availableProcessors() - 1);
            executor = new ParallelForestTraversalExecutor(treeUpdaters, threadPoolSize);
        } else {
            threadPoolSize = 0;
            executor = new SequentialForestTraversalExecutor(treeUpdaters);
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
     * @param randomSeed The random seed to use to create the forest random number generator
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
     * @return the number of points required by stream samplers before results are returned.
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
     * @return return the decay factor (lambda value) used by stream samplers in this forest.
     */
    public double getLambda() {
        return lambda;
    }

    /**
     * @return true if points are saved with sequence indexes, false otherwise.
     */
    public boolean storeSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    /**
     * @return true if tree nodes retain the center of mass, false otherwise.
     */
    public boolean centerOfMassEnabled() {
        return centerOfMassEnabled;
    }

    /**
     * @return true if parallel execution is enabled, false otherwise.
     */
    public boolean parallelExecutionEnabled() {
        return parallelExecutionEnabled;
    }

    /**
     * @return the number of threads in the thread pool if parallel execution is enabled, 0 otherwise.
     */
    public int getThreadPoolSize() {
        return threadPoolSize;
    }

    /**
     * Update the forest with the given point. The point is submitted to each sampler in the forest. If the sampler
     * accepts the point, the point is submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        executor.update(point);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into an aggregate
     * result. A visitor is constructed for each tree using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from all the trees are
     * combined using the accumulator and then transformed using the finisher before being returned. Trees are visited
     * in parallel using {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to construct a vistor.
     * @param accumulator    A function that combines the results from individual trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to produce the final result.
     * @param <R>            The visitor result type. This is the type that will be returned after traversing each individual
     *                       tree.
     * @param <S>            The final type, after any final normalization at the forest level.
     * @return The aggregated and finalized result after sending a visitor through each tree in the forest.
     */
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                                   BinaryOperator<R> accumulator, Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return executor.traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into an aggregate
     * result. A visitor is constructed for each tree using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from individual trees are
     * collected using the {@link java.util.stream.Collector} and returned. Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to construct a vistor.
     * @param collector      A collector used to aggregate individual tree results into a final result.
     * @param <R>            The visitor result type. This is the type that will be returned after traversing each individual
     *                       tree.
     * @param <S>            The final type, after any final normalization at the forest level.
     * @return The aggregated and finalized result after sending a visitor through each tree in the forest.
     */
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                                   Collector<R, ?, S> collector) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(collector, "collector must not be null");

        return executor.traverseForest(point, visitorFactory, collector);
    }

    /**
     * Visit each of the trees in the forest sequentially and combine the individual results into an aggregate
     * result. A visitor is constructed for each tree using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from all the trees are
     * combined using the {@link ConvergingAccumulator}, and the method stops visiting trees after convergence is
     * reached. The result is transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to construct a vistor.
     * @param accumulator    An accumulator that combines the results from individual trees into an aggregate result and
     *                       checks to see if the result can be returned without further processing.
     * @param finisher       A function called on the aggregate result in order to produce the final result.
     * @param <R>            The visitor result type. This is the type that will be returned after traversing each individual
     *                       tree.
     * @param <S>            The final type, after any final normalization at the forest level.
     * @return The aggregated and finalized result after sending a visitor through each tree in the forest.
     */
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                                   ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return executor.traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into an aggregate
     * result. A multi-visitor is constructed for each tree using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTreeMulti(double[], MultiVisitor)}. The results from all the
     * trees are combined using the accumulator and then transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to construct a multi-vistor.
     * @param accumulator    A function that combines the results from individual trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to produce the final result.
     * @param <R>            The visitor result type. This is the type that will be returned after traversing each individual
     *                       tree.
     * @param <S>            The final type, after any final normalization at the forest level.
     * @return The aggregated and finalized result after sending a visitor through each tree in the forest.
     */
    public <R, S> S traverseForestMulti(double[] point,
                                        Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
                                        BinaryOperator<R> accumulator, Function<R, S> finisher) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(accumulator, "accumulator must not be null");
        checkNotNull(finisher, "finisher must not be null");

        return executor.traverseForestMulti(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into an aggregate
     * result. A multi-visitor is constructed for each tree using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTreeMulti(double[], MultiVisitor)}. The results from individual
     * trees are collected using the {@link java.util.stream.Collector} and returned. Trees are visited in parallel
     * using {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to construct a vistor.
     * @param collector      A collector used to aggregate individual tree results into a final result.
     * @param <R>            The visitor result type. This is the type that will be returned after traversing each individual
     *                       tree.
     * @param <S>            The final type, after any final normalization at the forest level.
     * @return The aggregated and finalized result after sending a visitor through each tree in the forest.
     */
    public <R, S> S traverseForestMulti(double[] point,
                                        Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
                                        Collector<R, ?, S> collector) {

        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        checkNotNull(visitorFactory, "visitorFactory must not be null");
        checkNotNull(collector, "collector must not be null");

        return executor.traverseForestMulti(point, visitorFactory, collector);
    }

    /**
     * Compute an anomaly score for the given point. The point being scored is compared with the points in the sample
     * to compute a measure of how anomalous it is. Scores are greater than 0, with higher scores corresponding to
     * bing more anomalous. A threshold of 1.0 is commonly used to distinguish anomalous points from non-anomolous
     * ones.
     * <p>
     * See {@link AnomalyScoreVisitor} for more details about the anomaly score algorithm.
     *
     * @param point The point being scored.
     * @return an anomaly score for the given point.
     */
    public double getAnomalyScore(double[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        Function<RandomCutTree, Visitor<Double>> visitorFactory = tree ->
                new AnomalyScoreVisitor(point, tree.getRoot().getMass());

        BinaryOperator<Double> accumulator = Double::sum;

        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Anomaly score evaluated sequentially with option of early stopping the early stopping parameter precision gives an
     * approximate solution in the range (1-precision)*score(q)- precision, (1+precision)*score(q) + precision for the score of a point q.
     * In this function z is hardcoded to 0.1. If this function is used, then not all the trees will be used in
     * evaluation (but they have to be updated anyways, because they may be used for the next q). The advantage is that
     * "almost certainly" anomalies/non-anomalies can be detected easily with few trees.
     *
     * @param point input point q
     * @return anomaly score with early stopping with z=0.1
     */

    public double getApproximateAnomalyScore(double[] point) {
        if (!isOutputReady()) {
            return 0.0;
        }

        Function<RandomCutTree, Visitor<Double>> visitorFactory = tree ->
                new AnomalyScoreVisitor(point, tree.getRoot().getMass());

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED,
                numberOfTrees);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Compute an anomaly score attribution DiVector for the given point. The point being scored is compared with the points in the
     * sample to compute a measure of how anomalous it is. The result DiVector will contain an anomaly score in both
     * the positive and negative directions for each dimension of the data.
     * <p>
     * See {@link AnomalyAttributionVisitor} for more details about the anomaly score algorithm.
     *
     * @param point The point being scored.
     * @return an anomaly score for the given point.
     */
    public DiVector getAnomalyAttribution(double[] point) {
        // this will return the same (modulo floating point summation) L1Norm as getAnomalyScore
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        Function<RandomCutTree, Visitor<DiVector>> visitorFactory = tree ->
                new AnomalyAttributionVisitor(point, tree.getRoot().getMass());
        BinaryOperator<DiVector> accumulator = DiVector::addToLeft;
        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / numberOfTrees);

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Sequential version of attribution corresponding to getAnomalyScoreSequential; The high-low sum in the result
     * should be the same as the scalar score computed by {@link #getAnomalyScore(double[])}.
     *
     * @param point The point being scored.
     * @return anomaly attribution for the given point.
     */
    public DiVector getApproximateAnomalyAttribution(double[] point) {
        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        Function<RandomCutTree, Visitor<DiVector>> visitorFactory = tree ->
                new AnomalyAttributionVisitor(point, tree.getRoot().getMass());

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(
                dimensions,
                DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED,
                numberOfTrees);

        Function<DiVector, DiVector> finisher = vector -> vector.scale(1.0 / accumulator.getValuesAccepted());

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Compute a density estimate at the given point.
     * <p>
     * See {@link SimpleInterpolationVisitor} and {@link DensityOutput} for more details about the density computation.
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

        Function<RandomCutTree, Visitor<InterpolationMeasure>> visitorFactory = tree -> new SimpleInterpolationVisitor(point, sampleSize, 1.0, centerOfMassEnabled); //self
        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions, sampleSize, numberOfTrees);

        return new DensityOutput(traverseForest(point, visitorFactory, collector));
    }

    /**
     * Given a point with missing values, return a new point with the missing values imputed. Each tree in the forest
     * individual produces an imputed value. For 1-dimensional points, the median imputed value is returned. For
     * points with more than 1 dimension, the imputed point with the 25th percentile anomaly score is returned.
     *
     * @param point                 A point with missing values.
     * @param numberOfMissingValues The number of missing values in the point.
     * @param missingIndexes        An array containing the indexes of the missing values in the point. The length of the
     *                              array should be greater than or equal to the number of missing values.
     * @return A point with the missing values imputed.
     */
    public double[] imputeMissingValues(double[] point, int numberOfMissingValues, int[] missingIndexes) {
        checkArgument(numberOfMissingValues >= 0, "numberOfMissingValues must be greater than or equal to 0");

        // We check this condition in traverseForest, but we need to check it here s wellin case we need to copy the
        // point in the next block
        checkNotNull(point, "point must not be null");

        if (numberOfMissingValues == 0) {
            return Arrays.copyOf(point, point.length);
        }

        checkNotNull(missingIndexes, "missingIndexes must not be null");
        checkArgument(numberOfMissingValues <= missingIndexes.length,
            "numberOfMissingValues must be less than or equal to missingIndexes.length");

        if (!isOutputReady()) {
            return new double[dimensions];
        }

        Function<RandomCutTree, MultiVisitor<double[]>> visitorFactory = tree ->
            new ImputeVisitor(point, numberOfMissingValues, missingIndexes);

        if (numberOfMissingValues == 1) {

            // when there is 1 missing value, we sort all the imputed values and return the median

            Collector<double[], ArrayList<Double>, ArrayList<Double>> collector = Collector.of(
                ArrayList::new,
                (list, array) -> list.add(array[missingIndexes[0]]),
                (left, right) -> {
                    left.addAll(right);
                    return left;
                },
                list -> {
                    list.sort(Comparator.comparing(Double::doubleValue));
                    return list;
                }
            );

            ArrayList<Double> imputedValues = traverseForestMulti(point, visitorFactory, collector);
            double[] returnPoint = Arrays.copyOf(point, dimensions);
            returnPoint[missingIndexes[0]] = imputedValues.get(numberOfTrees / 2);
            return returnPoint;
        } else {

            // when there is more than 1 missing value, we sort the imputed points by anomaly score and
            // return the point with the 25th percentile anomaly score

            Collector<double[], ArrayList<double[]>, ArrayList<double[]>> collector = Collector.of(
                ArrayList::new,
                ArrayList::add,
                (left, right) -> {
                    left.addAll(right);
                    return left;
                },
                list -> {
                    list.sort(Comparator.comparing(this::getAnomalyScore));
                    return list;
                }
            );

            ArrayList<double[]> imputedPoints = traverseForestMulti(point, visitorFactory, collector);
            return imputedPoints.get(numberOfTrees / 4);
        }
    }

    /**
     * Given an initial shingled point, extrapolate the stream into the future to produce a forecast. This method is
     * intended to be called when the input data is being shingled, and it works by imputing forward one shingle
     * block at a time.
     *
     * @param point        The starting point for extrapolation.
     * @param horizon      The number of blocks to forecast.
     * @param blockSize    The number of entries in a block. This should be the same as the size of a single input to
     *                     the shingle.
     * @param cyclic       If true then the shingling is cyclic, otherwise it's a sliding shingle.
     * @param shingleIndex If cyclic is true, then this should be the current index in the shingle. That is, the index
     *                     where the next point added to the shingle would be written. If cyclic is false then this
     *                     value is not used.
     * @return a forecasted time series.
     */
    public double[] extrapolateBasic(double[] point, int horizon, int blockSize, boolean cyclic, int shingleIndex) {
        checkArgument(blockSize < dimensions && dimensions % blockSize == 0,
            "dimensions must be evenly divisible by blockSize");

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
     * Given an initial shingled point, extrapolate the stream into the future to produce a forecast. This method is
     * intended to be called when the input data is being shingled, and it works by imputing forward one shingle
     * block at a time. If the shingle is cyclic, then this method uses 0 as the shingle index.
     *
     * @param point     The starting point for extrapolation.
     * @param horizon   The number of blocks to forecast.
     * @param blockSize The number of entries in a block. This should be the same as the size of a single input to
     *                  the shingle.
     * @param cyclic    If true then the shingling is cyclic, otherwise it's a sliding shingle.
     * @return a forecasted time series.
     */
    public double[] extrapolateBasic(double[] point, int horizon, int blockSize, boolean cyclic) {
        return extrapolateBasic(point, horizon, blockSize, cyclic, 0);
    }

    /**
     * Given a shingle builder, extrapolate the stream into the future to produce a forecast. This method assumes you
     * are passing in the shingle builder used to preprocess points before adding them to this forest.
     *
     * @param builder The shingle builder used to process points before adding them to the forest.
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
                result[resultIndex++] = queryPoint[dimensions - blockSize + y] = imputedPoint[dimensions - blockSize + y];
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
                missingIndexes[y] = (currentPosition - blockSize + y) % dimensions;
            }

            double[] imputedPoint = imputeMissingValues(queryPoint, blockSize, missingIndexes);

            for (int y = 0; y < blockSize; y++) {
                result[resultIndex++] = queryPoint[(currentPosition - blockSize + y) % dimensions] =
                    imputedPoint[(currentPosition - blockSize + y) % dimensions];
            }

            // We want currentPosition - blockSize + y to be a valid index (which should be a positive multiple of blockSize)
            // When currentPosition == dimensions - blockSize, the following expression will be equal to dimensions

            currentPosition = (currentPosition + blockSize - 1) % dimensions + 1;
        }
    }

    /**
     * For each tree in the forest, follow the tree traversal path and return the leaf node if the standard Euclidean
     * distance between the query point and the leaf point is smaller than the given threshold. Note that this will not
     * necessarily be the nearest point in the tree, because the traversal path is determined by the random cuts in the
     * tree. If the same leaf point is found in multiple trees, those results will be combined into a single Neighbor
     * in the result.
     *
     * If sequence indexes are disabled for this forest, then the list of sequence indexes will be empty in returned
     * Neighbors.
     *
     * @param point             A point whose neighbors we want to find.
     * @param distanceThreshold The maximum Euclidean distance for a point to be considered a neighbor.
     * @return a list of Neighbors, ordered from closest to furthest.
     */
    public List<Neighbor> getNearNeighborsInSample(double[] point, double distanceThreshold) {
        checkNotNull(point, "point must not be null");
        checkArgument(distanceThreshold > 0, "distanceThreshold must be greater than 0");

        if (!isOutputReady()) {
            return Collections.emptyList();
        }

        Function<RandomCutTree, Visitor<Optional<Neighbor>>> visitorFactory = tree ->
            new NearNeighborVisitor(point, distanceThreshold);

        return traverseForest(point, visitorFactory, Neighbor.collector());
    }

    /**
     * For each tree in the forest, follow the tree traversal path and return the leaf node. Note that this will not
     * necessarily be the nearest point in the tree, because the traversal path is determined by the random cuts in the
     * tree. If the same leaf point is found in multiple trees, those results will be combined into a single Neighbor
     * in the result.
     *
     * If sequence indexes are disabled for this forest, then sequenceIndexes will be empty in the returned
     * Neighbors.
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
        return executor.getTotalUpdates() >= outputAfter;
    }

    /**
     * @return true if all samplers in the forest are full.
     */
    public boolean samplersFull() {
        return executor.getTotalUpdates() >= sampleSize;
    }

    /**
     * Returns the total number updates to the forest.
     *
     * The count of updates is represented with long type and may overflow.
     *
     * @return the total number of updates to the forest.
     */
    public long getTotalUpdates() {
        return executor.getTotalUpdates();
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make sense to use a constant default.

        private int dimensions;
        private int sampleSize = DEFAULT_SAMPLE_SIZE;
        private Optional<Integer> outputAfter = Optional.empty();
        private int numberOfTrees = DEFAULT_NUMBER_OF_TREES;
        private double lambda = DEFAULT_LAMBDA;
        private Optional<Long> randomSeed = Optional.empty();
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        private boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        private boolean parallelExecutionEnabled = DEFAULT_PARALLEL_EXECUTION_ENABLED;
        private Optional<Integer> threadPoolSize = Optional.empty();

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

        public T lambda(double lambda) {
            this.lambda = lambda;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = Optional.of(randomSeed);
            return (T) this;
        }

        public T windowSize(int windowSize) {
            this.lambda = 1.0 / windowSize;
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

        public RandomCutForest build() {
            return new RandomCutForest(this);
        }
    }
}
