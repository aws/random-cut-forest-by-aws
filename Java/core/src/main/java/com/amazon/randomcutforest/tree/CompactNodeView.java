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

import java.util.Arrays;
import java.util.Set;

public class CompactNodeView implements INodeView {
    final AbstractCompactRandomCutTree tree;
    int currentNodeOffset;

    public CompactNodeView(AbstractCompactRandomCutTree tree, int initialNodeIndex) {
        this.tree = tree;
        this.currentNodeOffset = initialNodeIndex;
    }

    public void setCurrentNodeIndex(int newOffset) {
        currentNodeOffset = newOffset;
    }

    public int getMass() {
        return tree.getMass(currentNodeOffset);
    }

    public BoundingBox getBoundingBox() {
        return tree.getBoundingBox(currentNodeOffset);
    }

    public BoundingBox getSiblingBoundingBox(double[] point) {
        if (tree.leftOf(point, currentNodeOffset)) {
            return tree.getBoundingBox(tree.internalNodes.rightIndex[currentNodeOffset]);
        } else {
            return tree.getBoundingBox(tree.internalNodes.leftIndex[currentNodeOffset]);
        }
    }

    public boolean leafPointEquals(double[] point) {
        return Arrays.equals(getLeafPoint(), point);
    }

    public int getCutDimension() {
        return tree.internalNodes.cutDimension[currentNodeOffset];
    }

    public double[] getLeafPoint() {
        return tree.getLeafPoint(currentNodeOffset);
    }

    public Set<Long> getSequenceIndexes() {
        return null;
    }

}
