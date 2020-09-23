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

import java.util.ArrayList;
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

import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.tree.CompactRandomCutTree;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.tree.SamplerPlusTree;

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
        AbstractForestUpdateExecutor<?> executor;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpExecutor() {

            int sampleSize = RandomCutForest.DEFAULT_SAMPLE_SIZE;
            double lambda = 1.0 / (sampleSize * RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_LAMBDA);
            int threadPoolSize = 4;
            Random random = new Random();

            if (!compactEnabled) {
                IUpdateCoordinator<double[]> updateCoordinator = new PointSequencer();
                ArrayList<IUpdatable<double[]>> trees = new ArrayList<>();
                for (int i = 0; i < numberOfTrees; i++) {
                    RandomCutTree tree = RandomCutTree.builder().build();
                    SimpleStreamSampler<double[]> sampler = new SimpleStreamSampler(sampleSize, lambda,
                            random.nextLong(), false);
                    SamplerPlusTree<double[]> samplingTree = new SamplerPlusTree(sampler, tree);
                    trees.add(samplingTree);
                }

                if (parallelExecutionEnabled) {
                    executor = new ParallelForestUpdateExecutor<>(updateCoordinator, trees, threadPoolSize);
                } else {
                    executor = new SequentialForestUpdateExecutor<>(updateCoordinator, trees);
                }
            } else {
                PointStore store = new PointStore(dimensions, numberOfTrees * sampleSize);
                IUpdateCoordinator<Integer> updateCoordinator = new PointStoreCoordinator(store);
                ArrayList<IUpdatable<Integer>> trees = new ArrayList<>();
                for (int i = 0; i < numberOfTrees; i++) {
                    CompactRandomCutTree tree = new CompactRandomCutTree(sampleSize, random.nextLong(), store);
                    SimpleStreamSampler<Integer> sampler = new SimpleStreamSampler(sampleSize, lambda,
                            random.nextLong(), false);
                    SamplerPlusTree<Integer> samplerTree = new SamplerPlusTree(sampler, tree);
                    trees.add(samplerTree);
                }

                if (parallelExecutionEnabled) {
                    executor = new ParallelForestUpdateExecutor(updateCoordinator, trees, threadPoolSize);
                } else {
                    executor = new SequentialForestUpdateExecutor(updateCoordinator, trees);
                }
            }
        }
    }

    private AbstractForestUpdateExecutor<?> executor;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public AbstractForestUpdateExecutor<?> updateOnly(BenchmarkState state) {
        double[][] data = state.data;
        executor = state.executor;

        for (int i = 0; i < data.length; i++) {
            executor.update(data[i]);
        }

        return executor;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public AbstractForestUpdateExecutor<?> updateAndGetAnomalyScore(BenchmarkState state, Blackhole blackhole) {
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
