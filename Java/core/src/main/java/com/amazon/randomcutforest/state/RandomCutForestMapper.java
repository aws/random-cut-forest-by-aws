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
import static com.amazon.randomcutforest.CommonUtils.validateInternalState;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.sampler.IStreamSampler;
import com.amazon.randomcutforest.state.sampler.CompactSamplerMapper;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreMapper;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeContext;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeState;
import com.amazon.randomcutforest.state.tree.RandomCutTreeMapper;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;

/**
 * A utility class for creating a {@link RandomCutForestState} instance from a
 * {@link RandomCutForest} instance and vice versa.
 */
@Getter
@Setter
public class RandomCutForestMapper
        implements IContextualStateMapper<RandomCutForest, RandomCutForestState, ExecutionContext> {

    /**
     * A flag indicating whether the structure of the trees in the forest should be
     * included in the state object. If true, then data describing the bounding
     * boxes and cuts defining each tree will be written to the
     * {@link RandomCutForestState} object produced by the mapper. Tree state is not
     * saved by default.
     */
    private boolean saveTreeStateEnabled = false;

    /**
     * A flag indicating whether the point store should be included in the
     * {@link RandomCutForestState} object produced by the mapper. This is saved by
     * default for compact trees
     */
    private boolean saveCoordinatorStateEnabled = true;

    /**
     * A flag indicating whether the samplers should be included in the
     * {@link RandomCutForestState} object produced by the mapper. This is saved by
     * default for all trees.
     */
    private boolean saveSamplerStateEnabled = true;

    /**
     * A flag indicating whether the executor context should be included in the
     * {@link RandomCutForestState} object produced by the mapper. Executor context
     * is not saved by default.
     */
    private boolean saveExecutorContextEnabled = false;

    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    /**
     * if true would require that the samplers populate the trees before the trees
     * can be used gain. That would correspond to extra time, at the benefit of a
     * smaller serialization.
     */
    private boolean partialTreeStateEnabled = false;

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

        RandomCutForestState state = new RandomCutForestState();

        state.setNumberOfTrees(forest.getNumberOfTrees());
        state.setDimensions(forest.getDimensions());
        state.setTimeDecay(forest.getTimeDecay());
        state.setSampleSize(forest.getSampleSize());
        state.setShingleSize(forest.getShingleSize());
        state.setCenterOfMassEnabled(forest.isCenterOfMassEnabled());
        state.setOutputAfter(forest.getOutputAfter());
        state.setStoreSequenceIndexesEnabled(forest.isStoreSequenceIndexesEnabled());
        state.setTotalUpdates(forest.getTotalUpdates());
        state.setCompact(true);
        state.setInternalShinglingEnabled(forest.isInternalShinglingEnabled());
        state.setBoundingBoxCacheFraction(forest.getBoundingBoxCacheFraction());
        state.setSaveSamplerStateEnabled(saveSamplerStateEnabled);
        state.setSaveTreeStateEnabled(saveTreeStateEnabled);
        state.setSaveCoordinatorStateEnabled(saveCoordinatorStateEnabled);
        state.setPrecision(forest.getPrecision().name());
        state.setCompressed(compressionEnabled);
        state.setPartialTreeState(partialTreeStateEnabled);
        state.setCurrentlySampling(forest.isCurrentlySampling());

        if (saveExecutorContextEnabled) {
            ExecutionContext executionContext = new ExecutionContext();
            executionContext.setParallelExecutionEnabled(forest.isParallelExecutionEnabled());
            executionContext.setThreadPoolSize(forest.getThreadPoolSize());
            state.setExecutionContext(executionContext);
        }

        if (saveCoordinatorStateEnabled) {
            PointStoreCoordinator<?> pointStoreCoordinator = (PointStoreCoordinator<?>) forest.getUpdateCoordinator();
            PointStoreMapper mapper = new PointStoreMapper();
            mapper.setCompressionEnabled(compressionEnabled);
            mapper.setNumberOfTrees(forest.getNumberOfTrees());
            PointStoreState pointStoreState = mapper.toState((PointStore) pointStoreCoordinator.getStore());
            state.setPointStoreState(pointStoreState);
        }
        List<CompactSamplerState> samplerStates = null;
        if (saveSamplerStateEnabled) {
            samplerStates = new ArrayList<>();
        }
        List<ITree<Integer, ?>> trees = null;
        if (saveTreeStateEnabled) {
            trees = new ArrayList<>();
        }

        CompactSamplerMapper samplerMapper = new CompactSamplerMapper();
        samplerMapper.setCompressionEnabled(compressionEnabled);

        for (IComponentModel<?, ?> component : forest.getComponents()) {
            SamplerPlusTree<Integer, ?> samplerPlusTree = (SamplerPlusTree<Integer, ?>) component;
            CompactSampler sampler = (CompactSampler) samplerPlusTree.getSampler();
            if (samplerStates != null) {
                samplerStates.add(samplerMapper.toState(sampler));
            }
            if (trees != null) {
                trees.add(samplerPlusTree.getTree());
            }
        }

        state.setCompactSamplerStates(samplerStates);

        if (trees != null) {
            RandomCutTreeMapper treeMapper = new RandomCutTreeMapper();
            List<CompactRandomCutTreeState> treeStates = trees.stream().map(t -> treeMapper.toState((RandomCutTree) t))
                    .collect(Collectors.toList());
            state.setCompactRandomCutTreeStates(treeStates);
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
     * @param state            A Random Cut Forest state object.
     * @param executionContext An executor context that will be used to initialize
     *                         new executors in the Random Cut Forest. If this
     *                         argument is null, then the mapper will look for an
     *                         executor context in the state object.
     * @param seed             A random seed.
     * @return A Random Cut Forest corresponding to the state object.
     * @throws NullPointerException if both the {@code executorContext} method
     *                              argument and the executor context field in the
     *                              state object are null.
     */
    public RandomCutForest toModel(RandomCutForestState state, ExecutionContext executionContext, long seed) {

        ExecutionContext ec;
        if (executionContext != null) {
            ec = executionContext;
        } else {
            checkNotNull(state.getExecutionContext(),
                    "The executor context in the state object is null, an executor context must be passed explicitly to toModel()");
            ec = state.getExecutionContext();
        }

        RandomCutForest.Builder<?> builder = RandomCutForest.builder().numberOfTrees(state.getNumberOfTrees())
                .dimensions(state.getDimensions()).timeDecay(state.getTimeDecay()).sampleSize(state.getSampleSize())
                .centerOfMassEnabled(state.isCenterOfMassEnabled()).outputAfter(state.getOutputAfter())
                .parallelExecutionEnabled(ec.isParallelExecutionEnabled()).threadPoolSize(ec.getThreadPoolSize())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).shingleSize(state.getShingleSize())
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction())
                .internalShinglingEnabled(state.isInternalShinglingEnabled()).randomSeed(seed);

        validateInternalState(Precision.valueOf(state.getPrecision()) == Precision.FLOAT_32,
                " use randomcutforest-serialization package");
        Random random = builder.getRandom();
        ComponentList<Integer, float[]> components = new ComponentList<>();
        CompactRandomCutTreeContext context = new CompactRandomCutTreeContext();
        IPointStore<Integer, float[]> pointStore = new PointStoreMapper().toModel(state.getPointStoreState());
        PointStoreCoordinator<float[]> coordinator = new PointStoreCoordinator<>(pointStore);
        coordinator.setTotalUpdates(state.getTotalUpdates());
        context.setPointStore(pointStore);
        context.setMaxSize(state.getSampleSize());
        RandomCutTreeMapper treeMapper = new RandomCutTreeMapper();
        List<CompactRandomCutTreeState> treeStates = state.isSaveTreeStateEnabled()
                ? state.getCompactRandomCutTreeStates()
                : null;
        CompactSamplerMapper samplerMapper = new CompactSamplerMapper();
        checkArgument(state.isSaveSamplerStateEnabled(), "samplers are not saved; no forest to reconstruct");
        List<CompactSamplerState> samplerStates = state.getCompactSamplerStates();
        for (int i = 0; i < state.getNumberOfTrees(); i++) {
            IStreamSampler<Integer> sampler = samplerMapper.toModel(samplerStates.get(i), random.nextLong());

            ITree<Integer, float[]> tree;
            if (treeStates != null) {
                tree = treeMapper.toModel(treeStates.get(i), context, random.nextLong());
                sampler.getSample().forEach(s -> tree.addPointToPartialTree(s.getValue(), s.getSequenceIndex()));
                tree.setConfig(Config.BOUNDING_BOX_CACHE_FRACTION, treeStates.get(i).getBoundingBoxCacheFraction());
                tree.validateAndReconstruct();
            } else {
                // using boundingBoxCache for the new tree
                tree = new RandomCutTree.Builder().capacity(state.getSampleSize()).randomSeed(random.nextLong())
                        .pointStoreView(pointStore).boundingBoxCacheFraction(state.getBoundingBoxCacheFraction())
                        .centerOfMassEnabled(state.isCenterOfMassEnabled())
                        .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).build();
                sampler.getSample().forEach(s -> tree.addPoint(s.getValue(), s.getSequenceIndex()));
            }
            components.add(new SamplerPlusTree<>(sampler, tree));
        }

        RandomCutForest forest = new RandomCutForest(builder, coordinator, components, random);
        if (!state.isCurrentlySampling()) {
            forest.pauseSampling();
        }
        return forest;

    }

    /**
     * Create a {@link RandomCutForest} instance from a {@link RandomCutForestState}
     * using the executor context in the state object. See
     * {@link #toModel(RandomCutForestState, ExecutionContext, long)}.
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
     * {@link #toModel(RandomCutForestState, ExecutionContext, long)}.
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
