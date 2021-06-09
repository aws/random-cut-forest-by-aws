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

import java.util.function.BiFunction;

import com.amazon.randomcutforest.tree.ITree;

/**
 * This is the interface for a visitor which can be used with
 * {RandomCutTree::traversePathToLeafAndVisitNodesMulti}. In this traversal
 * method, we optionally choose to split the visitor into two copies when
 * visiting nodes. Each copy then visits one of the paths down from that node.
 * The results from both visitors are combined before returning back up the
 * tree.
 */

public class MultiVisitorFactory<R> implements IMultiVisitorFactory<R> {
    private final BiFunction<ITree<?, ?>, double[], MultiVisitor<R>> newVisitor;
    private final BiFunction<ITree<?, ?>, R, R> liftResult;

    public MultiVisitorFactory(BiFunction<ITree<?, ?>, double[], MultiVisitor<R>> newVisitor,
            BiFunction<ITree<?, ?>, R, R> liftResult) {
        this.newVisitor = newVisitor;
        this.liftResult = liftResult;
    }

    public MultiVisitorFactory(BiFunction<ITree<?, ?>, double[], MultiVisitor<R>> newVisitor) {
        this(newVisitor, (tree, x) -> x);
    }

    @Override
    public MultiVisitor<R> newVisitor(ITree<?, ?> tree, double[] point) {
        return newVisitor.apply(tree, point);
    }

    @Override
    public R liftResult(ITree<?, ?> tree, R result) {
        return liftResult.apply(tree, result);
    }
}
