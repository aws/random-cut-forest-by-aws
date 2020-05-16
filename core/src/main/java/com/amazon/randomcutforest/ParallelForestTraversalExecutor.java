/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.RandomCutTree;

/**
 * An implementation of forest traversal methods that uses a private thread pool
 * to visit trees in parallel.
 */
public class ParallelForestTraversalExecutor extends AbstractForestTraversalExecutor {

    private ForkJoinPool forkJoinPool;
    private final int threadPoolSize;

    public ParallelForestTraversalExecutor(ArrayList<TreeUpdater> treeUpdaters, int threadPoolSize) {
        super(treeUpdaters);
        this.threadPoolSize = threadPoolSize;
        forkJoinPool = new ForkJoinPool(threadPoolSize);
    }

    @Override
    protected void update(double[] pointCopy, long sequenceIndex) {
        submitAndJoin(() -> {
            treeUpdaters.parallelStream().forEach(updater -> updater.update(pointCopy, sequenceIndex));
            return null;
        });
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> treeUpdaters.parallelStream().map(TreeUpdater::getTree).map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(() -> treeUpdaters.parallelStream().map(TreeUpdater::getTree).map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).collect(collector));
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (int i = 0; i < treeUpdaters.size(); i += threadPoolSize) {
            final int start = i;
            final int end = Math.min(start + threadPoolSize, treeUpdaters.size());

            List<R> results = submitAndJoin(
                    () -> treeUpdaters.subList(start, end).parallelStream().map(TreeUpdater::getTree).map(tree -> {
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
    public <R, S> S traverseForestMulti(double[] point, Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> treeUpdaters.parallelStream().map(TreeUpdater::getTree).map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(() -> treeUpdaters.parallelStream().map(TreeUpdater::getTree).map(tree -> {
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
