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

package com.amazon.randomcutforest.executor;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IMultiVisitorFactory;
import com.amazon.randomcutforest.IVisitorFactory;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;

/**
 * An implementation of forest traversal methods that uses a private thread pool
 * to visit trees in parallel.
 */
public class ParallelForestTraversalExecutor extends AbstractForestTraversalExecutor {

    private ForkJoinPool forkJoinPool;
    private final int threadPoolSize;

    public ParallelForestTraversalExecutor(ComponentList<?, ?> treeExecutors, int threadPoolSize) {
        super(treeExecutors);
        this.threadPoolSize = threadPoolSize;
        forkJoinPool = new ForkJoinPool(threadPoolSize);
    }

    @Override
    public <R, S> S traverseForest(double[] point, IVisitorFactory<R> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher) {

        return submitAndJoin(() -> components.parallelStream().map(c -> c.traverse(point, visitorFactory))
                .reduce(accumulator).map(finisher))
                        .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForest(double[] point, IVisitorFactory<R> visitorFactory, Collector<R, ?, S> collector) {

        return submitAndJoin(
                () -> components.parallelStream().map(c -> c.traverse(point, visitorFactory)).collect(collector));
    }

    @Override
    public <R, S> S traverseForest(double[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (int i = 0; i < components.size(); i += threadPoolSize) {
            final int start = i;
            final int end = Math.min(start + threadPoolSize, components.size());

            List<R> results = submitAndJoin(() -> components.subList(start, end).parallelStream()
                    .map(c -> c.traverse(point, visitorFactory)).collect(Collectors.toList()));
            results.forEach(accumulator::accept);

            if (accumulator.isConverged()) {
                break;
            }
        }

        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, IMultiVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> components.parallelStream().map(c -> c.traverseMulti(point, visitorFactory))
                .reduce(accumulator).map(finisher))
                        .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(
                () -> components.parallelStream().map(c -> c.traverseMulti(point, visitorFactory)).collect(collector));
    }

    private <T> T submitAndJoin(Callable<T> callable) {
        if (forkJoinPool == null) {
            forkJoinPool = new ForkJoinPool(threadPoolSize);
        }
        return forkJoinPool.submit(callable).join();
    }
}
