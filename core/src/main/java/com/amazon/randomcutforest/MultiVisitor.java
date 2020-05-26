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

import com.amazon.randomcutforest.tree.Node;

/**
 * This is the interface for a visitor which can be used with
 * {RandomCutTree::traversePathToLeafAndVisitNodesMulti}. In this traversal
 * method, we optionally choose to split the visitor into two copies when
 * visiting nodes. Each copy then visits one of the paths down from that node.
 * The results from both visitors are combined before returning back up the
 * tree.
 */
public interface MultiVisitor<R> extends Visitor<R> {

    /**
     * Returns true of the traversal method should split the visitor (i.e., create a
     * copy) at this node.
     *
     * @param node A node in the tree traversal
     * @return true if the traversal should split the visitor into two copies at
     *         this node, false otherwise.
     */
    boolean trigger(final Node node);

    /**
     * Return a copy of this visitor. The original visitor plus the copy will each
     * traverse one branch of the tree.
     *
     * @return a copy of this visitor
     */
    MultiVisitor<R> newCopy();

    /**
     * Combine two visitors. The state of the argument visitor should be combined
     * with the state of this instance. This method is called after both visitors
     * have traversed one branch of the tree.
     *
     * @param other A second visitor
     */
    void combine(MultiVisitor<R> other);
}
