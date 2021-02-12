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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.profilers.OutputSizeProfiler;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.protostuff.LinkedBuffer;
import io.protostuff.ProtostuffIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;

@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(value = 1)
@State(Scope.Benchmark)
public class StateMapperBenchmark {
    public static final int NUM_TRAIN_SAMPLES = 2048;
    public static final int NUM_TEST_SAMPLES = 50;

    @State(Scope.Thread)
    public static class BenchmarkState {
        @Param({ "10" })
        int dimensions;

        @Param({ "50" })
        int numberOfTrees;

        @Param({ "256" })
        int sampleSize;

        @Param({ "false", "true" })
        boolean saveTreeState;

        @Param({ "SINGLE", "DOUBLE" })
        Precision precision;

        double[][] trainingData;
        double[][] testData;
        RandomCutForestState forestState;
        String json;
        byte[] protostuff;

        @Setup(Level.Trial)
        public void setUpData() {
            NormalMixtureTestData gen = new NormalMixtureTestData();
            trainingData = gen.generateTestData(NUM_TRAIN_SAMPLES, dimensions);
            testData = gen.generateTestData(NUM_TEST_SAMPLES, dimensions);
        }

        @Setup(Level.Invocation)
        public void setUpForest() throws JsonProcessingException {
            RandomCutForest forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions)
                    .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();

            for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
                forest.update(trainingData[i]);
            }

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            mapper.setSaveTreeState(saveTreeState);
            forestState = mapper.toState(forest);

            ObjectMapper jsonMapper = new ObjectMapper();
            json = jsonMapper.writeValueAsString(forestState);

            Schema<RandomCutForestState> schema = RuntimeSchema.getSchema(RandomCutForestState.class);
            LinkedBuffer buffer = LinkedBuffer.allocate(512);
            try {
                protostuff = ProtostuffIOUtil.toByteArray(forestState, schema, buffer);
            } finally {
                buffer.clear();
            }
        }
    }

    private byte[] bytes;

    @TearDown(Level.Iteration)
    public void tearDown() {
        OutputSizeProfiler.setTestArray(bytes);
    }

    @Benchmark
    @OperationsPerInvocation(NUM_TEST_SAMPLES)
    public RandomCutForestState roundTripFromState(BenchmarkState state, Blackhole blackhole) {
        RandomCutForestState forestState = state.forestState;
        double[][] testData = state.testData;

        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            mapper.setSaveTreeState(state.saveTreeState);
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
    public String roundTripFromJson(BenchmarkState state, Blackhole blackhole) throws JsonProcessingException {
        String json = state.json;
        double[][] testData = state.testData;

        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            ObjectMapper jsonMapper = new ObjectMapper();
            RandomCutForestState forestState = jsonMapper.readValue(json, RandomCutForestState.class);

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            mapper.setSaveTreeState(state.saveTreeState);
            RandomCutForest forest = mapper.toModel(forestState);

            double score = forest.getAnomalyScore(testData[i]);
            blackhole.consume(score);
            forest.update(testData[i]);
            json = jsonMapper.writeValueAsString(mapper.toState(forest));
        }

        bytes = json.getBytes();
        return json;
    }

    @Benchmark
    @OperationsPerInvocation(NUM_TEST_SAMPLES)
    public byte[] roundTripFromProtostuff(BenchmarkState state, Blackhole blackhole) {
        bytes = state.protostuff;
        double[][] testData = state.testData;

        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            Schema<RandomCutForestState> schema = RuntimeSchema.getSchema(RandomCutForestState.class);
            RandomCutForestState forestState = schema.newMessage();
            ProtostuffIOUtil.mergeFrom(bytes, forestState, schema);

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContext(true);
            mapper.setSaveTreeState(state.saveTreeState);
            RandomCutForest forest = mapper.toModel(forestState);

            double score = forest.getAnomalyScore(testData[i]);
            blackhole.consume(score);
            forest.update(testData[i]);
            forestState = mapper.toState(forest);

            LinkedBuffer buffer = LinkedBuffer.allocate(512);
            try {
                bytes = ProtostuffIOUtil.toByteArray(forestState, schema, buffer);
            } finally {
                buffer.clear();
            }
        }

        return bytes;
    }
}
