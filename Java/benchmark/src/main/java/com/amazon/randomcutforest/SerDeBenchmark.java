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
import org.openjdk.jmh.infra.Blackhole;

import com.amazon.randomcutforest.serialize.RandomCutForestSerDe;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(value = 1)
@State(Scope.Benchmark)
public class SerDeBenchmark {
    public static final int NUM_TRAIN_SAMPLES = 2048;
    public static final int NUM_TEST_SAMPLES = 50;

    @State(Scope.Thread)
    public static class BenchmarkState {
        @Param({ "10" })
        int dimensions;

        @Param({ "100" })
        int numberOfTrees;

        @Param({ "256" })
        int sampleSize;

        double[][] trainingData;
        double[][] testData;
        RandomCutForestState forestState;
        String json;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData gen = new NormalMixtureTestData();
            trainingData = gen.generateTestData(NUM_TRAIN_SAMPLES, dimensions);
            testData = gen.generateTestData(NUM_TEST_SAMPLES, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpForest() {
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).numberOfTrees(numberOfTrees)
                    .sampleSize(sampleSize).build();

            for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
                forest.update(trainingData[i]);
            }

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            forestState = mapper.toState(forest);

            RandomCutForestSerDe serDe = new RandomCutForestSerDe();
            serDe.getMapper().setSaveTreeState(true);
            json = serDe.toJson(forest);
        }
    }

    @Benchmark
    @OperationsPerInvocation(NUM_TEST_SAMPLES)
    public RandomCutForestState roundTripFromState(BenchmarkState state, Blackhole blackhole) {
        RandomCutForestState forestState = state.forestState;
        double[][] testData = state.testData;

        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            RandomCutForest forest = mapper.toModel(forestState);
            double score = forest.getAnomalyScore(testData[i]);
            blackhole.consume(score);
            forest.update(testData[i]);
            forestState = mapper.toState(forest);
        }

        return forestState;
    }

    @Benchmark
    @OperationsPerInvocation(NUM_TEST_SAMPLES)
    public String roundTripFromJson(BenchmarkState state, Blackhole blackhole) {
        String json = state.json;
        double[][] testData = state.testData;

        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            RandomCutForestSerDe serDe = new RandomCutForestSerDe();
            serDe.getMapper().setSaveExecutorContext(true);
            RandomCutForest forest = serDe.fromJson(json);
            double score = forest.getAnomalyScore(testData[i]);
            blackhole.consume(score);
            forest.update(testData[i]);
            json = serDe.toJson(forest);
        }

        return json;
    }
}
