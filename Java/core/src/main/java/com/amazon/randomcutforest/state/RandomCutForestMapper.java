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

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.RandomCutForest;
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
import com.amazon.randomcutforest.state.store.PointStoreDoubleState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeContext;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeMapper;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeState;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
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
     * {@link RandomCutForestState} object produced by the mapper.
     */
    private boolean saveTreeState;

    /**
     * A flag indicating whether the executor context should be included in the
     * {@link RandomCutForestState} object produced by the mapper.
     */
    private boolean saveExecutorContext;

    /**
     * Crate a new mapper with {@code saveTreeState} and {@code saveExecutorContext}
     * set to false.
     */
    public RandomCutForestMapper() {
        saveTreeState = false;
        saveExecutorContext = false;
    }

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

        if (saveExecutorContext) {
            ExecutorContext executorContext = new ExecutorContext();
            executorContext.setParallelExecutionEnabled(forest.isParallelExecutionEnabled());
            executorContext.setThreadPoolSize(forest.getThreadPoolSize());
            state.setExecutorContext(executorContext);
        }

        if (forest.isCompactEnabled()) {
            PointStoreCoordinator pointStoreCoordinator = (PointStoreCoordinator) forest.getUpdateCoordinator();
            PointStoreDoubleState pointStoreState = new PointStoreDoubleMapper()
                    .toState(pointStoreCoordinator.getStore());
            state.setPointStoreDoubleState(pointStoreState);

            List<CompactSamplerState> samplerStates = new ArrayList<>();
            List<CompactRandomCutTreeState> treeStates = null;
            if (saveTreeState) {
                treeStates = new ArrayList<>();
            }

            CompactSamplerMapper samplerMapper = new CompactSamplerMapper();
            CompactRandomCutTreeMapper treeMapper = new CompactRandomCutTreeMapper();

            for (IComponentModel<?> component : forest.getComponents()) {
                SamplerPlusTree<Integer> samplerPlusTree = (SamplerPlusTree<Integer>) component;
                CompactSampler sampler = (CompactSampler) samplerPlusTree.getSampler();
                samplerStates.add(samplerMapper.toState(sampler));
                if (treeStates != null) {
                    CompactRandomCutTreeDouble tree = (CompactRandomCutTreeDouble) samplerPlusTree.getTree();
                    treeStates.add(treeMapper.toState(tree));
                }
            }
            state.setCompactSamplerStates(samplerStates);
            state.setCompactRandomCutTreeStates(treeStates);
        } else {
            ArraySamplersToCompactStateConverter converter = new ArraySamplersToCompactStateConverter(
                    forest.isStoreSequenceIndexesEnabled(), forest.getDimensions(),
                    forest.getNumberOfTrees() * forest.getSampleSize());

            for (IComponentModel<?> model : forest.getComponents()) {
                SamplerPlusTree<double[]> samplerPlusTree = (SamplerPlusTree<double[]>) model;
                SimpleStreamSampler<double[]> sampler = (SimpleStreamSampler<double[]>) samplerPlusTree.getSampler();
                converter.addSampler(sampler);
            }

            state.setPointStoreDoubleState(converter.getPointStoreDoubleState());
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
        PointStoreDouble pointStore = new PointStoreDoubleMapper().toModel(state.getPointStoreDoubleState());

        if (state.isCompactEnabled()) {
            PointStoreCoordinator coordinator = new PointStoreCoordinator(pointStore);
            coordinator.setTotalUpdates(state.getTotalUpdates());

            ComponentList<Integer> components = new ComponentList<>();
            CompactRandomCutTreeMapper treeMapper = new CompactRandomCutTreeMapper();

            CompactRandomCutTreeContext context = new CompactRandomCutTreeContext();
            context.setMaxSize(state.getSampleSize());
            context.setPointStore(pointStore);

            for (int i = 0; i < state.getNumberOfTrees(); i++) {
                CompactRandomCutTreeDouble tree;
                if (treeStates != null) {
                    tree = treeMapper.toModel(treeStates.get(i), context, rng.nextLong());
                } else {
                    tree = new CompactRandomCutTreeDouble(state.getSampleSize(), rng.nextLong(), pointStore,
                            state.isBoundingBoxCachingEnabled(), state.isCenterOfMassEnabled(),
                            state.isStoreSequenceIndexesEnabled());
                }

                CompactSampler sampler = samplerMapper.toModel(samplerStates.get(i), state, rng.nextLong());

                if (treeStates == null) {
                    sampler.getSample().forEach(s -> tree.addPoint(s.getValue(), s.getSequenceIndex()));
                }

                components.add(new SamplerPlusTree<>(sampler, tree));
            }

            return new RandomCutForest(builder, coordinator, components, rng);
        } else {

            PassThroughCoordinator coordinator = new PassThroughCoordinator();
            ComponentList<double[]> components = new ComponentList<>();

            for (int i = 0; i < state.getNumberOfTrees(); i++) {
                CompactSampler compactData = samplerMapper.toModel(samplerStates.get(i), state);
                RandomCutTree tree = RandomCutTree.builder()
                        .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled())
                        .centerOfMassEnabled(state.isCenterOfMassEnabled()).randomSeed(rng.nextLong()).build();
                SimpleStreamSampler<double[]> sampler = new SimpleStreamSampler<>(state.getSampleSize(),
                        state.getLambda(), rng.nextLong());

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
