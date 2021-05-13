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


package com.amazon.randomcutforest.profilers;


import java.util.Collection;
import java.util.Collections;
import org.openjdk.jmh.infra.BenchmarkParams;
import org.openjdk.jmh.infra.IterationParams;
import org.openjdk.jmh.profile.InternalProfiler;
import org.openjdk.jmh.results.AggregationPolicy;
import org.openjdk.jmh.results.IterationResult;
import org.openjdk.jmh.results.Result;
import org.openjdk.jmh.results.ScalarResult;

/**
 * This simple profile outputs the size of a provided byte array or string as part of the JMH
 * metrics. We use it to measure the size of output in {@link
 * com.amazon.randomcutforest.StateMapperBenchmark}.
 */
public class OutputSizeProfiler implements InternalProfiler {

    private static byte[] bytes;

    public static void setTestString(String s) {
        bytes = s.getBytes();
    }

    public static void setTestArray(byte[] bytes) {
        OutputSizeProfiler.bytes = bytes;
    }

    @Override
    public void beforeIteration(BenchmarkParams benchmarkParams, IterationParams iterationParams) {}

    @Override
    public Collection<? extends Result> afterIteration(
            BenchmarkParams benchmarkParams,
            IterationParams iterationParams,
            IterationResult iterationResult) {
        int length = 0;
        if (bytes != null) {
            length = bytes.length;
            bytes = null;
        }
        ScalarResult result =
                new ScalarResult("+output-size.bytes", length, "bytes", AggregationPolicy.AVG);
        return Collections.singleton(result);
    }

    @Override
    public String getDescription() {
        return null;
    }
}
