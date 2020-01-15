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

import java.util.Arrays;

import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.NormalMixtureTestData;
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

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Benchmark)
public class TreeUpdaterBenchmark {

    public static final int DATA_SIZE = 50_000;

    @State(Scope.Benchmark)
    public static class BenchmarkState {
        @Param({"1", "16", "256"})
        int dimensions;

        @Param({"0.0", "1e-5"})
        double lambda;

        @Param({"false", "true"})
        boolean storeSequenceIndexesEnabled;

        double[][] data;
        TreeUpdater treeUpdater;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Iteration)
        public void setUpTree() {
            SimpleStreamSampler sampler = new SimpleStreamSampler(RandomCutForest.DEFAULT_SAMPLE_SIZE, lambda, 99);
            RandomCutTree tree = RandomCutTree.builder()
                    .randomSeed(101)
                    .storeSequenceIndexesEnabled(storeSequenceIndexesEnabled)
                    .build();
            treeUpdater = new TreeUpdater(sampler, tree);
        }
    }

    private TreeUpdater tree;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public TreeUpdater update(BenchmarkState state) {
        double[][] data = state.data;
        tree = state.treeUpdater;
        long entriesSeen = 0;

        for (int i = 0; i < data.length; i++) {
            try {
                tree.update(data[i], ++entriesSeen);
            } catch (Exception e) {
                System.out.println("Point = " + Arrays.toString(data[i]));
                System.out.println("Sequence Index = " + entriesSeen);
                throw e;
            }
        }

        return tree;
    }
}
