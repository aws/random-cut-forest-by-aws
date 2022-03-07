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

import com.amazon.randomcutforest.tree.INodeView;

/**
 * This is the interface for a visitor which can be used to query a ITraversable
 * to produce a result. A visitor is submitted to
 * ITraversable#traverse(double[], Visitor), and during the traversal the
 * {@link #acceptLeaf} and {@link #accept} methods are invoked on the nodes in
 * the traversal path.
 * <p>
 * See ITraversable#traverse(double[], Visitor) for details about the traversal
 * path.
 */
public interface Visitor<R> {
    /**
     * Visit a node in the traversal path.
     *
     * @param node        the node being visited
     * @param depthOfNode the depth of the node being visited
     */
    void accept(INodeView node, int depthOfNode);

    /**
     * Visit the leaf node in the traversal path. By default, this method proxies to
     * {@link #accept(INodeView, int)}.
     *
     * @param leafNode    the leaf node being visited
     * @param depthOfNode the depth of the leaf node
     */
    default void acceptLeaf(INodeView leafNode, final int depthOfNode) {
        accept(leafNode, depthOfNode);
    }

    /**
     * At the end of the traversal, this method is called to obtain the result
     * computed by the visitor.
     *
     * @return the result value computed by the visitor.
     */
    R getResult();

    /**
     * This method short-circuits the evaluation of the Visitor at nodes on the
     * traversal path. By default, the accept (or acceptLeaf) method will be invoked
     * for each Node in the traversal path. But the NodeView has to prepare
     * information to support that visitor invocation. Before invocation, the value
     * of isConverged will be checked. If it is true, some of that preparation can
     * be skipped -- because the visitor would not be updated. This method can be
     * overwritten to optimize visitors that do not need to visit every node on the
     * root to leaf path before returning a value.
     *
     * Mote that this convergence applies to a single visitor computation and is
     * expected to be a speedup without any change in the value of the answer. This
     * is different from converging accumulator which corresponds to sequential
     * evaluation of different visitors and early stopping.
     **/
    default boolean isConverged() {
        return false;
    }
}
