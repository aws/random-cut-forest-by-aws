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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.IUpdatableTree;

/**
 * An implementation of forest traversal methods that uses a private thread pool
 * to visit trees in parallel.
 */
public class ParallelForestTraversalExecutorV2<P> extends GenericForestTraversalExecutor<P> {

    private ForkJoinPool forkJoinPool;
    private final int threadPoolSize;

    public ParallelForestTraversalExecutorV2(IUpdateCoordinator<P> updateCoordinator,
            ArrayList<IUpdatableTree<P>> trees, int threadPoolSize) {
        super(updateCoordinator, trees);
        this.threadPoolSize = threadPoolSize;
        forkJoinPool = new ForkJoinPool(threadPoolSize);
    }

    @Override
    protected List<P> update(P point) {
        return submitAndJoin(() -> trees.parallelStream().map(t -> t.update(point)).filter(Objects::nonNull)
                .collect(Collectors.toList()));
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> trees.parallelStream().map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(() -> trees.parallelStream().map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).collect(collector));
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (int i = 0; i < trees.size(); i += threadPoolSize) {
            final int start = i;
            final int end = Math.min(start + threadPoolSize, trees.size());

            List<R> results = submitAndJoin(() -> trees.subList(start, end).parallelStream().map(tree -> {
                Visitor<R> visitor = visitorFactory.apply(tree);
                return tree.traverseTree(point, visitor);
            }).collect(Collectors.toList()));

            results.forEach(accumulator::accept);

            if (accumulator.isConverged()) {
                break;
            }
        }

        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> trees.parallelStream().map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(() -> trees.parallelStream().map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).collect(collector));
    }

    private <T> T submitAndJoin(Callable<T> callable) {
        if (forkJoinPool == null) {
            forkJoinPool = new ForkJoinPool(threadPoolSize);
        }
        return forkJoinPool.submit(callable).join();
    }
}
