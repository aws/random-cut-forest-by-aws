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

import java.util.List;
import java.util.Random;

import org.github.jamm.MemoryMeter;
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

import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(value = 1)
@State(Scope.Thread)
public class RandomCutForestBenchmark {

    public final static int DATA_SIZE = 50_000;
    public final static int INITIAL_DATA_SIZE = 25_000;

    @State(Scope.Benchmark)
    public static class BenchmarkState {
        @Param({ "40" })
        int baseDimensions;

        @Param({ "1" })
        int shingleSize;

        @Param({ "30" })
        int numberOfTrees;

        @Param({ "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.0" })
        double boundingBoxCacheFraction;

        @Param({ "false", "true" })
        boolean parallel;

        double[][] data;
        RandomCutForest forest;

        @Setup(Level.Trial)
        public void setUpData() {
            int dimensions = baseDimensions * shingleSize;
            NormalMixtureTestData gen = new NormalMixtureTestData();
            data = gen.generateTestData(INITIAL_DATA_SIZE + DATA_SIZE, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpForest() {
            forest = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(baseDimensions * shingleSize)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).parallelExecutionEnabled(parallel)
                    .boundingBoxCacheFraction(boundingBoxCacheFraction).randomSeed(99).build();

            for (int i = 0; i < INITIAL_DATA_SIZE; i++) {
                forest.update(data[i]);
            }
        }
    }

    private RandomCutForest forest;

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest updateOnly(BenchmarkState state) {
        double[][] data = state.data;
        forest = state.forest;

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
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

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
            score += forest.getAnomalyScore(data[i]);
            if (rnd.nextDouble() < 0.01) {
                forest.update(data[i]); // this should execute sparingly
            }
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

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
            score = forest.getAnomalyScore(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(score);
        if (!forest.parallelExecutionEnabled) {
            MemoryMeter meter = new MemoryMeter();
            System.out.println(" forest size " + meter.measureDeep(forest));
        }
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest attributionAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        DiVector vector = new DiVector(forest.getDimensions());

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
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

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
            output = forest.getSimpleDensity(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(output);
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest basicNeighborAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        List<Neighbor> output = null;

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
            output = forest.getNearNeighborsInSample(data[i]);
            forest.update(data[i]);
        }

        blackhole.consume(output);
        return forest;
    }

    @Benchmark
    @OperationsPerInvocation(DATA_SIZE)
    public RandomCutForest imputeAndUpdate(BenchmarkState state, Blackhole blackhole) {
        double[][] data = state.data;
        forest = state.forest;
        double[] output = null;

        for (int i = INITIAL_DATA_SIZE; i < data.length; i++) {
            output = forest.imputeMissingValues(data[i], 1, new int[] { forest.dimensions - 1 });
            forest.update(data[i]);
        }

        blackhole.consume(output);
        return forest;
    }
}
