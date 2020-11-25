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

import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Thread)
public class RandomCutForestBenchmark {

    public final static int DATA_SIZE = 50_000;

    @State(Scope.Benchmark)
    public static class BenchmarkState {
        @Param({ "10" })
        int dimensions;

        @Param({ "100" })
        int numberOfTrees;

        @Param({ "false", "true" })
        boolean parallelExecutionEnabled;

        double[][] data;
        RandomCutForest forest;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData testData = new NormalMixtureTestData();
            data = testData.generateTestData(DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpForest() {
            forest = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions)
                    .parallelExecutionEnabled(parallelExecutionEnabled).compactEnabled(true).randomSeed(99).build();
        }
    }

    private RandomCutForest forest;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest updateOnly(BenchmarkState state) {
        double[][] data = state.data;
        forest = state.forest;

        for (int i = 0; i < data.length; i++) {
            forest.update(data[i]);
        }

        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest scoreOnly(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        double score = 0.0;
        Random rnd = new Random(0);

        for (int i = 0; i < data.length; i++) {
            score += forest.getAnomalyScore(data[i]);
            if (rnd.nextDouble() < 0.01)
                forest.update(data[i]); // this should execute sparingly
        }

        blackhole.consume(score);
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest scoreAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        double score = 0.0;

        for (int i = 0; i < data.length; i++) {
            score = forest.getAnomalyScore(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(score);
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest attributionAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        DiVector vector = new DiVector(forest.getDimensions());

        for (int i = 0; i < data.length; i++) {
            vector = forest.getAnomalyAttribution(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(vector);
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest basicDensityAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        DensityOutput output = new DensityOutput(forest.getDimensions(), forest.getSampleSize());

        for (int i = 0; i < data.length; i++) {
            output = forest.getSimpleDensity(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(output);
        return forest;
    }
}
