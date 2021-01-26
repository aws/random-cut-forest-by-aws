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
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.profilers.StringSizeProfiler;
import com.amazon.randomcutforest.serialize.RandomCutForestSerDe;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

/**
 * The main purpose of this benchmark is to measure the length of the JSON
 * strings produced by
 * {@link com.amazon.randomcutforest.serialize.RandomCutForestSerDe} under
 * different configuration options. The {@link SerDeBenchmark} will give a
 * better measurement of the speed of serialization and deserialization, because
 * it performs more measurements in a loop.
 */
@Warmup(iterations = 1)
@Measurement(iterations = 3)
@Fork(value = 1)
@State(Scope.Benchmark)
public class SerDeOutputLengthBenchmark {

    @State(Scope.Thread)
    public static class BenchmarkState {
        @Param({ "1", "4", "10" })
        int dimensions;

        @Param({ "50" })
        int numberOfTrees;

        @Param({ "128", "256" })
        int sampleSize;

        @Param({ "SINGLE", "DOUBLE" })
        Precision precision;

        RandomCutForest forest;
        RandomCutForestSerDe serDe;

        @Setup(Level.Invocation)
        public void setUpForest() {
            forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions).numberOfTrees(numberOfTrees)
                    .sampleSize(sampleSize).precision(precision).build();

            NormalMixtureTestData gen = new NormalMixtureTestData();
            double[][] trainingData = gen.generateTestData(sampleSize * 10, dimensions);

            for (double[] point : trainingData) {
                forest.update(point);
            }

            serDe = new RandomCutForestSerDe();
        }
    }

    private String json;

    @TearDown(Level.Iteration)
    public void tearDown() {
        StringSizeProfiler.setTestString(json);
    }

    @Benchmark
    public String outputLengthWithTreeState(BenchmarkState state) {
        state.serDe.getMapper().setSaveTreeState(true);
        json = state.serDe.toJson(state.forest);
        return json;
    }

    @Benchmark
    public String outputLengthWithoutTreeState(BenchmarkState state) {
        state.serDe.getMapper().setSaveTreeState(false);
        json = state.serDe.toJson(state.forest);
        return json;
    }
}
