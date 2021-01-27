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

import com.amazon.randomcutforest.SerDeOutputLengthBenchmark;

/**
 * This simple profile outputs the size of a provided string in bytes as part of
 * the JMH metrics. We use it to measure the size of JSON output in
 * {@link SerDeOutputLengthBenchmark}.
 */
public class StringSizeProfiler implements InternalProfiler {

    private static String testString;

    public static void setTestString(String s) {
        testString = s;
    }

    @Override
    public void beforeIteration(BenchmarkParams benchmarkParams, IterationParams iterationParams) {
    }

    @Override
    public Collection<? extends Result> afterIteration(BenchmarkParams benchmarkParams, IterationParams iterationParams,
            IterationResult iterationResult) {
        int length = 0;
        if (testString != null) {
            length = testString.getBytes().length;
            testString = null;
        }
        ScalarResult result = new ScalarResult("+string-size.stringSize", length, "bytes", AggregationPolicy.AVG);
        return Collections.singleton(result);
    }

    @Override
    public String getDescription() {
        return null;
    }
}
