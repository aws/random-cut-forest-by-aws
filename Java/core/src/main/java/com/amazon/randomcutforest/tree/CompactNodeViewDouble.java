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

public class CompactNodeViewDouble implements INodeView<double[]> {
    final CompactRandomCutTreeDouble treeRef;
    short currentNodeOffset;

    public CompactNodeViewDouble(short nodeId, CompactRandomCutTreeDouble treeRef) {
        this.treeRef = treeRef;
        this.currentNodeOffset = nodeId;
    }

    public void updateNode(short newId) {
        currentNodeOffset = newId;
    }

    public int getMass() {
        return treeRef.getMass(currentNodeOffset);
    }

    public BoundingBox getBoundingBox() {
        return treeRef.getBoundingBox(currentNodeOffset);
    }

    public BoundingBox getSiblingBoundingBox(double[] point) {
        if (treeRef.leftOf(point, currentNodeOffset)) {
            return treeRef.getBoundingBox(treeRef.internalNodes.rightIndex[currentNodeOffset]);
        } else {
            return treeRef.getBoundingBox(treeRef.internalNodes.leftIndex[currentNodeOffset]);
        }
    }

    public boolean leafPointEquals(double[] point) {
        return Arrays.equals(getLeafPoint(), point);
    }

    public int getCutDimension() {
        return treeRef.internalNodes.cutDimension[currentNodeOffset];
    }

    public double[] getLeafPoint() {
        return treeRef.getLeafPoint(currentNodeOffset);
    }

    public Set<Long> getSequenceIndexes() {
        return null;
    }

}
