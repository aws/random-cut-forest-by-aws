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

package com.amazon.randomcutforest.state;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.PassThroughCoordinator;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.state.sampler.ArraySamplersToCompactStateConverter;
import com.amazon.randomcutforest.state.sampler.CompactSamplerMapper;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleMapper;
import com.amazon.randomcutforest.state.store.PointStoreFloatMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeContext;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeDoubleMapper;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeFloatMapper;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeState;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;

/**
 * A utility class for creating a {@link RandomCutForestState} instance from a
 * {@link RandomCutForest} instance and vice versa.
 */
@Getter
@Setter
public class RandomCutForestMapper
        implements IContextualStateMapper<RandomCutForest, RandomCutForestState, ExecutorContext> {

    /**
     * A flag indicating whether the structure of the trees in the forest should be
     * included in the state object. If true, then data describing the bounding
     * boxes and cuts defining each tree will be written to the
     * {@link RandomCutForestState} object produced by the mapper. Tree state is not
     * saved by default.
     */
    private boolean saveTreeState = false;

    /**
     * A flag indicating whether the executor context should be included in the
     * {@link RandomCutForestState} object produced by the mapper. Executor context
     * is not saved by defalt.
     */
    private boolean saveExecutorContext = false;

    /**
     * If true, then model data will be copied (i.e., the state class will not share
     * any data with the model). If false, some model data may be shared with the
     * state class. Copying is enabled by default.
     */
    private boolean copy = true;

    /**
     * Create a {@link RandomCutForestState} object representing the state of the
     * given forest. If the forest is compact and the {@code saveTreeState} flag is
     * set to true, then structure of the trees in the forest will be included in
     * the state object. If the flag is set to false, then the state object will
     * only contain the sampler data for each tree. If the
     * {@code saveExecutorContext} is true, then the executor context will be
     * included in the state object.
     *
     * @param forest A Random Cut Forest whose state we want to capture.
     * @return a {@link RandomCutForestState} object representing the state of the
     *         given forest.
     * @throws IllegalArgumentException if the {@code saveTreeState} flag is true
     *                                  and the forest is not compact.
     */
    @Override
    public RandomCutForestState toState(RandomCutForest forest) {
        if (saveTreeState) {
            checkArgument(forest.isCompactEnabled(), "tree state cannot be saved for noncompact forests");
        }

        RandomCutForestState state = new RandomCutForestState();
        state.setNumberOfTrees(forest.getNumberOfTrees());
        state.setDimensions(forest.getDimensions());
        state.setLambda(forest.getLambda());
        state.setSampleSize(forest.getSampleSize());
        state.setCenterOfMassEnabled(forest.isCenterOfMassEnabled());
        state.setOutputAfter(forest.getOutputAfter());
        state.setStoreSequenceIndexesEnabled(forest.isStoreSequenceIndexesEnabled());
        state.setTotalUpdates(forest.getTotalUpdates());
        state.setCompactEnabled(forest.isCompactEnabled());
        state.setBoundingBoxCachingEnabled(forest.isBoundingBoxCachingEnabled());

        if (saveExecutorContext) {
            ExecutorContext executorContext = new ExecutorContext();
            executorContext.setParallelExecutionEnabled(forest.isParallelExecutionEnabled());
            executorContext.setThreadPoolSize(forest.getThreadPoolSize());
            state.setExecutorContext(executorContext);
        }

        if (forest.isCompactEnabled()) {
            PointStoreCoordinator pointStoreCoordinator = (PointStoreCoordinator) forest.getUpdateCoordinator();
            PointStoreState pointStoreState;
            if (forest.getPrecision() == Precision.SINGLE) {
                pointStoreState = new PointStoreFloatMapper()
                        .toState((PointStoreFloat) pointStoreCoordinator.getStore());
            } else {
                pointStoreState = new PointStoreDoubleMapper()
                        .toState((PointStoreDouble) pointStoreCoordinator.getStore());
            }
            state.setPointStoreState(pointStoreState);

            List<CompactSamplerState> samplerStates = new ArrayList<>();
            List<ITree<Integer, ?>> trees = null;
            if (saveTreeState) {
                trees = new ArrayList<>();
            }

            CompactSamplerMapper samplerMapper = new CompactSamplerMapper();

            for (IComponentModel<?, ?> component : forest.getComponents()) {
                SamplerPlusTree<Integer, ?> samplerPlusTree = (SamplerPlusTree<Integer, ?>) component;
                CompactSampler sampler = (CompactSampler) samplerPlusTree.getSampler();
                samplerStates.add(samplerMapper.toState(sampler));
                if (trees != null) {
                    trees.add(samplerPlusTree.getTree());
                }
            }

            state.setCompactSamplerStates(samplerStates);

            if (trees != null) {
                if (forest.getPrecision() == Precision.SINGLE) {
                    CompactRandomCutTreeFloatMapper treeMapper = new CompactRandomCutTreeFloatMapper();
                    List<CompactRandomCutTreeState> treeStates = trees.stream()
                            .map(t -> treeMapper.toState((CompactRandomCutTreeFloat) t)).collect(Collectors.toList());
                    state.setCompactRandomCutTreeStates(treeStates);
                } else {
                    CompactRandomCutTreeDoubleMapper treeMapper = new CompactRandomCutTreeDoubleMapper();
                    List<CompactRandomCutTreeState> treeStates = trees.stream()
                            .map(t -> treeMapper.toState((CompactRandomCutTreeDouble) t)).collect(Collectors.toList());
                    state.setCompactRandomCutTreeStates(treeStates);
                }
            }
        } else {
            ArraySamplersToCompactStateConverter converter = new ArraySamplersToCompactStateConverter(
                    forest.isStoreSequenceIndexesEnabled(), forest.getDimensions(),
                    forest.getNumberOfTrees() * forest.getSampleSize());

            for (IComponentModel<?, ?> model : forest.getComponents()) {
                SamplerPlusTree<double[], ?> samplerPlusTree = (SamplerPlusTree<double[], ?>) model;
                SimpleStreamSampler<double[]> sampler = (SimpleStreamSampler<double[]>) samplerPlusTree.getSampler();
                converter.addSampler(sampler);
            }

            state.setPointStoreState(converter.getPointStoreDoubleState());
            state.setCompactSamplerStates(converter.getCompactSamplerStates());
        }

        return state;
    }

    /**
     * Create a {@link RandomCutForest} instance from a
     * {@link RandomCutForestState}. If the state contains tree states, then trees
     * will be constructed from the tree state objects. Otherwise, empty trees are
     * created and populated from the sampler data. The resulting forest should be
     * equal in distribution to the forest that the state object was created from.
     *
     * @param state           A Random Cut Forest state object.
     * @param executorContext An executor context that will be used to initialize
     *                        new executors in the Random Cut Forest. If this
     *                        argument is null, then the mapper will look for an
     *                        executor context in the state object.
     * @param seed            A random seed.
     * @return A Random Cut Forest corresponding to the state object.
     * @throws NullPointerException if both the {@code executorContext} method
     *                              argument and the executor context field in the
     *                              state object are null.
     */
    @Override
    public RandomCutForest toModel(RandomCutForestState state, ExecutorContext executorContext, long seed) {

        ExecutorContext ec;
        if (executorContext != null) {
            ec = executorContext;
        } else {
            checkNotNull(state.getExecutorContext(),
                    "The executor context in the state object is null, an executor context must be passed explicitly to toModel()");
            ec = state.getExecutorContext();
        }

        RandomCutForest.Builder<?> builder = RandomCutForest.builder().numberOfTrees(state.getNumberOfTrees())
                .dimensions(state.getDimensions()).lambda(state.getLambda()).sampleSize(state.getSampleSize())
                .centerOfMassEnabled(state.isCenterOfMassEnabled()).outputAfter(state.getOutputAfter())
                .parallelExecutionEnabled(ec.isParallelExecutionEnabled()).threadPoolSize(ec.getThreadPoolSize())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled())
                .boundingBoxCachingEnabled(state.isBoundingBoxCachingEnabled())
                .compactEnabled(state.isCompactEnabled());

        Random rng = builder.getRandom();

        List<CompactRandomCutTreeState> treeStates = state.getCompactRandomCutTreeStates();
        List<CompactSamplerState> samplerStates = state.getCompactSamplerStates();
        CompactSamplerMapper samplerMapper = new CompactSamplerMapper();

        if (state.isCompactEnabled()) {
            PointStoreState pointStoreState = state.getPointStoreState();
            CompactRandomCutTreeContext context = new CompactRandomCutTreeContext();
            context.setMaxSize(state.getSampleSize());

            if (pointStoreState.isSinglePrecisionSet()) {
                ComponentList<Integer, float[]> components = new ComponentList<>();
                IPointStore<float[]> pointStore = new PointStoreFloatMapper().toModel(pointStoreState);
                PointStoreCoordinator<float[]> coordinator = new PointStoreCoordinator<>(pointStore);
                coordinator.setTotalUpdates(state.getTotalUpdates());
                context.setPointStore(pointStore);
                CompactRandomCutTreeFloatMapper treeMapper = new CompactRandomCutTreeFloatMapper();
                treeMapper.setBoundingBoxCacheEnabled(state.isBoundingBoxCachingEnabled());
                for (int i = 0; i < state.getNumberOfTrees(); i++) {
                    ITree<Integer, float[]> tree;
                    if (treeStates != null) {
                        tree = treeMapper.toModel(treeStates.get(i), context, rng.nextLong());
                    } else {
                        tree = new CompactRandomCutTreeFloat(state.getSampleSize(), rng.nextLong(), pointStore,
                                state.isBoundingBoxCachingEnabled(), state.isCenterOfMassEnabled(),
                                state.isStoreSequenceIndexesEnabled());
                    }
                    CompactSampler sampler = samplerMapper.toModel(samplerStates.get(i), rng.nextLong());
                    if (treeStates == null) {
                        sampler.getSample().forEach(s -> tree.addPoint(s.getValue(), s.getSequenceIndex()));
                    }
                    components.add(new SamplerPlusTree<>(sampler, tree));
                }
                builder.precision(Precision.SINGLE);
                return new RandomCutForest(builder, coordinator, components, rng);
            } else {
                ComponentList<Integer, double[]> components = new ComponentList<>();
                IPointStore<double[]> pointStore = new PointStoreDoubleMapper().toModel(pointStoreState);
                PointStoreCoordinator<double[]> coordinator = new PointStoreCoordinator<>(pointStore);
                coordinator.setTotalUpdates(state.getTotalUpdates());
                context.setPointStore(pointStore);
                CompactRandomCutTreeDoubleMapper treeMapper = new CompactRandomCutTreeDoubleMapper();
                treeMapper.setBoundingBoxCacheEnabled(state.isBoundingBoxCachingEnabled());
                for (int i = 0; i < state.getNumberOfTrees(); i++) {
                    ITree<Integer, double[]> tree;
                    if (treeStates != null) {
                        tree = treeMapper.toModel(treeStates.get(i), context, rng.nextLong());
                    } else {
                        tree = new CompactRandomCutTreeDouble(state.getSampleSize(), rng.nextLong(), pointStore,
                                state.isBoundingBoxCachingEnabled(), state.isCenterOfMassEnabled(),
                                state.isStoreSequenceIndexesEnabled());
                    }
                    CompactSampler sampler = samplerMapper.toModel(samplerStates.get(i), rng.nextLong());
                    if (treeStates == null) {
                        sampler.getSample().forEach(s -> tree.addPoint(s.getValue(), s.getSequenceIndex()));
                    }
                    components.add(new SamplerPlusTree<>(sampler, tree));
                }
                builder.precision(Precision.DOUBLE);
                return new RandomCutForest(builder, coordinator, components, rng);
            }
        } else {
            PointStoreDouble pointStore = new PointStoreDoubleMapper().toModel(state.getPointStoreState());
            PassThroughCoordinator coordinator = new PassThroughCoordinator();
            coordinator.setTotalUpdates(state.getTotalUpdates());
            ComponentList<double[], double[]> components = new ComponentList<>();
            for (int i = 0; i < state.getNumberOfTrees(); i++) {
                CompactSampler compactData = samplerMapper.toModel(samplerStates.get(i));
                RandomCutTree tree = RandomCutTree.builder()
                        .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled())
                        .centerOfMassEnabled(state.isCenterOfMassEnabled()).randomSeed(rng.nextLong()).build();
                SimpleStreamSampler<double[]> sampler = new SimpleStreamSampler<>(state.getSampleSize(),
                        state.getLambda(), rng.nextLong());
                sampler.setMaxSequenceIndex(compactData.getMaxSequenceIndex());
                sampler.setSequenceIndexOfMostRecentLambdaUpdate(
                        compactData.getSequenceIndexOfMostRecentLambdaUpdate());

                for (Weighted<Integer> sample : compactData.getWeightedSample()) {
                    double[] point = pointStore.get(sample.getValue());
                    sampler.addSample(new Weighted<>(point, sample.getWeight(), sample.getSequenceIndex()));
                    tree.addPoint(point, sample.getSequenceIndex());
                }
                components.add(new SamplerPlusTree<>(sampler, tree));
            }

            return new RandomCutForest(builder, coordinator, components, rng);
        }

    }

    /**
     * Create a {@link RandomCutForest} instance from a {@link RandomCutForestState}
     * using the executor context in the state object. See
     * {@link #toModel(RandomCutForestState, ExecutorContext, long)}.
     *
     * @param state A Random Cut Forest state object.
     * @param seed  A random seed.
     * @return A Random Cut Forest corresponding to the state object.
     * @throws NullPointerException if the executor context field in the state
     *                              object are null.
     */
    public RandomCutForest toModel(RandomCutForestState state, long seed) {
        return toModel(state, null, seed);
    }

    /**
     * Create a {@link RandomCutForest} instance from a {@link RandomCutForestState}
     * using the executor context in the state object. See
     * {@link #toModel(RandomCutForestState, ExecutorContext, long)}.
     *
     * @param state A Random Cut Forest state object.
     * @return A Random Cut Forest corresponding to the state object.
     * @throws NullPointerException if the executor context field in the state
     *                              object are null.
     */
    public RandomCutForest toModel(RandomCutForestState state) {
        return toModel(state, null);
    }
}
