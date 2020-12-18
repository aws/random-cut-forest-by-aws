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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.Arrays;
import java.util.Set;

public class CompactNodeView implements INodeView<Integer> {
    final AbstractCompactRandomCutTree<?> tree;
    int currentNodeOffset;

    public CompactNodeView(AbstractCompactRandomCutTree<?> tree, int initialNodeIndex) {
        this.tree = tree;
        this.currentNodeOffset = initialNodeIndex;
    }

    public int getMass() {
        return tree.getMass(currentNodeOffset);
    }

    public IBoundingBoxView getBoundingBox() {
        return tree.getBoundingBox(currentNodeOffset);
    }

    public IBoundingBoxView getSiblingBoundingBox(double[] point) {
        if (tree.leftOf(point, currentNodeOffset)) {
            return tree.getBoundingBox(tree.nodeManager.getRightChild(currentNodeOffset));
        } else {
            return tree.getBoundingBox(tree.nodeManager.getLeftChild(currentNodeOffset));
        }
    }

    public boolean leafPointEquals(double[] point) {
        return Arrays.equals(getLeafPoint(), point);
    }

    public int getCutDimension() {
        return tree.nodeManager.getCutDimension(currentNodeOffset);
    }

    public double[] getLeafPoint() {
        return tree.getPoint(currentNodeOffset);
    }

    public Set<Long> getSequenceIndexes() {
        return null;
    }

    @Override
    public INodeView<Integer> getNodeView(Integer node) {
        checkState(node != null, " incorrect node in nodeview");
        currentNodeOffset = node.intValue();
        return this;
    }

}
