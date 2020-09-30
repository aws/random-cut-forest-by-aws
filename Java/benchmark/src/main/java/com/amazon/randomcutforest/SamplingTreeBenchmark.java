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

import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.tree.RandomCutTree;

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Benchmark)
public class SamplingTreeBenchmark {

    public static final int DATA_SIZE = 50_000;

    @State(Scope.Thread)
    public static class BenchmarkState {
        @Param({ "1", "16", "256" })
        int dimensions;

        @Param({ "0.0", "1e-5" })
        double lambda;

        @Param({ "false", "true" })
        boolean storeSequenceIndexesEnabled;

        double[][] data;
        SamplerPlusTree<double[]> samplingTree;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpTree() {
            SimpleStreamSampler<double[]> sampler = new SimpleStreamSampler(RandomCutForest.DEFAULT_SAMPLE_SIZE, lambda,
                    99, storeSequenceIndexesEnabled);
            RandomCutTree tree = RandomCutTree.builder().randomSeed(101)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled).build();
            samplingTree = new SamplerPlusTree<>(sampler, tree);
        }
    }

    private SamplerPlusTree<double[]> samplerPlusTree;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public SamplerPlusTree<double[]> update(BenchmarkState state) {
        double[][] data = state.data;
        samplerPlusTree = state.samplingTree;
        long entriesSeen = 0;

        for (int i = 0; i < data.length; i++) {
            // Sequential<double[]> sequential = new Sequential(data[i], 0.0,
            // ++entriesSeen);
            samplerPlusTree.update(data[i], ++entriesSeen);
        }

        return samplerPlusTree;
    }
}
