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


import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.tree.ITree;
import java.util.function.Function;

/** This interface defines a model that can be traversed by a {@link Visitor}. */
public interface ITraversable {
    /**
     * Traverse the path defined by {@code point} and invoke the visitor. The path defined by {@code
     * point} is the path from the root node to the leaf node where {@code point} would be inserted.
     * The visitor is invoked for each node in the path in reverse order (starting from the leaf
     * node and ending at the root node). The return value is obtained by calling {@link
     * Visitor#getResult()} on the visitor after it has visited each node in the path.
     *
     * @param point A point that determines the traversal path.
     * @param visitorFactory A factory function that can be applied to an {@link ITree} instance to
     *     obtain a {@link Visitor} instance.
     * @param <R> The return value type of the visitor.
     * @return the value of {@link Visitor#getResult()} after visiting each node in the path.
     */
    <R> R traverse(double[] point, Function<ITree<?, ?>, Visitor<R>> visitorFactory);

    /**
     * Traverse the paths defined by {@code point} and the multi-visitor, and invoke the
     * multi-visitor on each node. The path defined by {@code point} is the path from the root node
     * to the leaf node where {@code point} would be inserted. However, at each node along the path
     * we invoke {@link MultiVisitor#trigger}, and if it returns true we create a copy of the
     * visitor and send it down both branches of the tree. The multi-visitor is invoked for each
     * node in the path in reverse order (starting from the leaf node and ending at the root node).
     * When two multi-visitors meet at a node, they are combined by calling {@link
     * MultiVisitor#combine}. The return value is obtained by calling {@link
     * MultiVisitor#getResult()} on the single remaining visitor after it has visited each node in
     * each branch the path.
     *
     * @param point A point that determines the traversal path.
     * @param visitorFactory A factory function that can be applied to an {@link ITree} instance to
     *     obtain a {@link MultiVisitor} instance.
     * @param <R> The return value type of the multi-visitor.
     * @return the value of {@link MultiVisitor#getResult()} after traversing all paths.
     */
    <R> R traverseMulti(double[] point, Function<ITree<?, ?>, MultiVisitor<R>> visitorFactory);

    /**
     * After a new traversable model is initialized, it will not be able to return meaningful
     * results to queries until it has been updated with (i.e., learned from) some number of points.
     * The exact number of points may vary for different models. After this method returns true for
     * the first time, it should continue to return true unless the user takes an explicit action to
     * reset the model state.
     *
     * @return true if this model is ready to provide a meaningful response to a traversal query,
     *     otherwise false.
     */
    boolean isOutputReady();
}
