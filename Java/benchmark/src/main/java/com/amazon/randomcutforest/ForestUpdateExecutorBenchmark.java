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

import java.util.Random;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import com.amazon.randomcutforest.executor.AbstractForestUpdateExecutor;
import com.amazon.randomcutforest.executor.IStateCoordinator;
import com.amazon.randomcutforest.executor.ParallelForestUpdateExecutor;
import com.amazon.randomcutforest.executor.PassThroughCoordinator;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.executor.SequentialForestUpdateExecutor;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;
import com.amazon.randomcutforest.tree.RandomCutTree;

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Thread)
public class ForestUpdateExecutorBenchmark {

    public final static int DATA_SIZE = 50_000;

    @State(Scope.Benchmark)
    public static class BenchmarkState {
        @Param({ "1", "16", "256" })
        int dimensions;

        @Param({ "50", "100" })
        int numberOfTrees;

        @Param({ "false", "true" })
        boolean parallelExecutionEnabled;

        @Param({ "false", "true" })
        boolean compactEnabled;

        double[][] data;
        AbstractForestUpdateExecutor<?, ?> executor;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpExecutor() {

            int sampleSize = RandomCutForest.DEFAULT_SAMPLE_SIZE;
            double lambda = 1.0 / (sampleSize * RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY);
            int threadPoolSize = 4;
            Random random = new Random();

            if (!compactEnabled) {
                IStateCoordinator<double[], double[]> updateCoordinator = new PassThroughCoordinator();
                ComponentList<double[], double[]> components = new ComponentList<>();
                for (int i = 0; i < numberOfTrees; i++) {
                    RandomCutTree tree = RandomCutTree.builder().build();
                    SimpleStreamSampler<double[]> sampler = SimpleStreamSampler.<double[]>builder().capacity(sampleSize)
                            .timeDecay(lambda).randomSeed(random.nextLong()).build();
                    SamplerPlusTree<double[], double[]> samplingTree = new SamplerPlusTree<>(sampler, tree);
                    components.add(samplingTree);
                }

                if (parallelExecutionEnabled) {
                    executor = new ParallelForestUpdateExecutor<>(updateCoordinator, components, threadPoolSize);
                } else {
                    executor = new SequentialForestUpdateExecutor<>(updateCoordinator, components);
                }
            } else {
                PointStoreDouble store = new PointStoreDouble(dimensions, numberOfTrees * sampleSize);
                IStateCoordinator<Integer, double[]> updateCoordinator = new PointStoreCoordinator(store);
                ComponentList<Integer, double[]> components = new ComponentList<>();
                for (int i = 0; i < numberOfTrees; i++) {
                    CompactRandomCutTreeDouble tree = new CompactRandomCutTreeDouble.Builder().maxSize(sampleSize)
                            .randomSeed(random.nextLong()).pointStore(store).boundingBoxCacheFraction(1.0)
                            .centerOfMassEnabled(false).storeSequenceIndexesEnabled(false).build();
                    SimpleStreamSampler<Integer> sampler = SimpleStreamSampler.<Integer>builder().capacity(sampleSize)
                            .timeDecay(lambda).randomSeed(random.nextLong()).build();
                    SamplerPlusTree<Integer, double[]> samplerTree = new SamplerPlusTree<>(sampler, tree);
                    components.add(samplerTree);
                }

                if (parallelExecutionEnabled) {
                    executor = new ParallelForestUpdateExecutor<>(updateCoordinator, components, threadPoolSize);
                } else {
                    executor = new SequentialForestUpdateExecutor<>(updateCoordinator, components);
                }
            }
        }
    }

    private AbstractForestUpdateExecutor<?, ?> executor;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public AbstractForestUpdateExecutor<?, ?> updateOnly(BenchmarkState state) {
        double[][] data = state.data;
        executor = state.executor;

        for (int i = 0; i < data.length; i++) {
            executor.update(data[i]);
        }

        return executor;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public AbstractForestUpdateExecutor<?, ?> updateAndGetAnomalyScore(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        executor = state.executor;
        double score = 0.0;

        int i;
        for (i = 0; i < RandomCutForest.DEFAULT_SAMPLE_SIZE; i++) {
            executor.update(data[i]);
        }

        for (; i < data.length; i++) {
            executor.update(data[i]);
        }

        blackhole.consume(score);
        return executor;
    }

}
