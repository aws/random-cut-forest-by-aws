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
import org.github.jamm.MemoryMeter;
import org.openjdk.jmh.infra.BenchmarkParams;
import org.openjdk.jmh.infra.IterationParams;
import org.openjdk.jmh.profile.InternalProfiler;
import org.openjdk.jmh.results.AggregationPolicy;
import org.openjdk.jmh.results.IterationResult;
import org.openjdk.jmh.results.Result;
import org.openjdk.jmh.results.ScalarResult;

/** A profiler that uses the JAMM memory meter to measure the size of an object graph. */
public class ObjectGraphSizeProfiler implements InternalProfiler {

    private static Object object;
    private static MemoryMeter meter = new MemoryMeter();

    public static void setObject(Object object) {
        ObjectGraphSizeProfiler.object = object;
    }

    @Override
    public void beforeIteration(BenchmarkParams benchmarkParams, IterationParams iterationParams) {}

    @Override
    public Collection<? extends Result> afterIteration(
            BenchmarkParams benchmarkParams,
            IterationParams iterationParams,
            IterationResult iterationResult) {
        long size = 0;
        if (object != null) {
            size = meter.measureDeep(object);
            object = null;
        }
        ScalarResult result =
                new ScalarResult("+object-graph-size.bytes", size, "bytes", AggregationPolicy.AVG);
        return Collections.singleton(result);
    }

    @Override
    public String getDescription() {
        return null;
    }
}
