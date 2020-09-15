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

import com.amazon.randomcutforest.sampler.SimpleStreamSamplerV2;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.tree.SamplingTree;

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Thread)
public class GenericForestTraversalExecutorBenchmark {

    public final static int DATA_SIZE = 50_000;

    @State(Scope.Benchmark)
    public static class BenchmarkState {
        @Param({ "1", "16", "256" })
        int dimensions;

        @Param({ "50", "100" })
        int numberOfTrees;

        @Param({ "false", "true" })
        boolean parallelExecutionEnabled;

        double[][] data;
        AbstractForestUpdateExecutor<Sequential<double[]>> executor;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpExecutor() {
            IUpdateCoordinator<Sequential<double[]>> updateCoordinator = new PointSequencer();
            ArrayList<IUpdatable<Sequential<double[]>>> trees = new ArrayList<>();
            Random random = new Random();
            for (int i = 0; i < numberOfTrees; i++) {
                RandomCutTree tree = RandomCutTree.builder().build();
                SimpleStreamSamplerV2<double[]> sampler = new SimpleStreamSamplerV2<>(double[].class,
                        RandomCutForest.DEFAULT_SAMPLE_SIZE,
                        1.0 / (RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_LAMBDA
                                * RandomCutForest.DEFAULT_SAMPLE_SIZE),
                        random.nextLong());
                SamplingTree<double[]> samplingTree = new SamplingTree<>(sampler, tree);
                trees.add(samplingTree);
            }

            if (parallelExecutionEnabled) {
                executor = new ParallelForestUpdateExecutor<>(updateCoordinator, trees, 4);
            } else {
                executor = new SequentialForestUpdateExecutor<>(updateCoordinator, trees);
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