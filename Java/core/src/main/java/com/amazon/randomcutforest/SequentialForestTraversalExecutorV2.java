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
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.IUpdatableTree;

/**
 * Traverse the trees in a forest sequentially.
 */
public class SequentialForestTraversalExecutorV2<P> extends GenericForestTraversalExecutor<P> {

    public SequentialForestTraversalExecutorV2(IUpdateCoordinator<P> updateCoordinator,
            ArrayList<IUpdatableTree<P>> trees) {
        super(updateCoordinator, trees);
    }

    @Override
    protected List<P> update(P point) {
        return trees.stream().map(t -> t.update(point)).filter(Objects::nonNull).collect(Collectors.toList());
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        R unnormalizedResult = trees.stream().map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).reduce(accumulator).orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return trees.stream().map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).collect(collector);
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (ITree<?> tree : trees) {
            Visitor<R> visitor = visitorFactory.apply(tree);
            accumulator.accept(tree.traverseTree(point, visitor));
            if (accumulator.isConverged()) {
                break;
            }
        }

        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        R unnormalizedResult = trees.stream().map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).reduce(accumulator).orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return trees.stream().map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).collect(collector);
    }
}
