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
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.RandomCutTree;

/**
 * Traverse the trees in a forest sequentially.
 */
public class SequentialForestTraversalExecutor extends AbstractForestTraversalExecutor {

    public SequentialForestTraversalExecutor(ArrayList<TreeUpdater> treeUpdaters) {
        super(treeUpdaters);
    }

    @Override
    protected void update(double[] pointCopy, long sequenceIndex) {
        treeUpdaters.forEach(updater -> {
            updater.update(pointCopy, sequenceIndex);
        });
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        R unnormalizedResult = treeUpdaters.stream().map(TreeUpdater::getTree).map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).reduce(accumulator).orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return treeUpdaters.stream().map(TreeUpdater::getTree).map(tree -> {
            Visitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTree(point, visitor);
        }).collect(collector);
    }

    @Override
    public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (TreeUpdater treeUpdater : treeUpdaters) {
            RandomCutTree tree = treeUpdater.getTree();
            Visitor<R> visitor = visitorFactory.apply(tree);
            accumulator.accept(tree.traverseTree(point, visitor));
            if (accumulator.isConverged()) {
                break;
            }
        }

        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        R unnormalizedResult = treeUpdaters.stream().map(TreeUpdater::getTree).map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).reduce(accumulator).orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForestMulti(double[] point, Function<RandomCutTree, MultiVisitor<R>> visitorFactory,
            Collector<R, ?, S> collector) {

        return treeUpdaters.stream().map(TreeUpdater::getTree).map(tree -> {
            MultiVisitor<R> visitor = visitorFactory.apply(tree);
            return tree.traverseTreeMulti(point, visitor);
        }).collect(collector);
    }
}
