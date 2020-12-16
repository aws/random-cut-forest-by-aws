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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;

public class CompactRandomCutTreeDouble extends AbstractCompactRandomCutTree<double[]> {

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore, boolean cacheEnabled) {
        super(maxSize, seed, cacheEnabled);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBox[maxSize - 1];
        }
    }

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore, LeafStore leafStore,
            NodeStore nodeStore, int rootIndex) {
        super(maxSize, seed, leafStore, nodeStore, rootIndex, true);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        cachedBoxes = new BoundingBox[maxSize - 1];
    }

    @Override
    String toString(double[] point) {
        return Arrays.toString(point);
    }

    @Override
    IBoundingBox<double[]> getLeafBoxFromPoint(int pointIndex) {
        return new BoundingBox(pointStore.get(pointIndex));
    }

    @Override
    IBoundingBox<double[]> getInternalMergedBox(double[] point, double[] oldPoint) {
        return BoundingBox.getMergedBox(point, oldPoint);
    }

    @Override
    boolean checkEqual(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean leftOf(double[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    @Override
    protected double[] getLeafPoint(int nodeOffset) {
        double[] internal = pointStore.get(leafNodes.getPointIndex(nodeOffset));
        return Arrays.copyOf(internal, internal.length);
    }

    /**
     * creates the bounding box of a node/leaf
     *
     * @param nodeReference node in question
     * @return the bounding box
     */

    @Override
    IBoundingBox<double[]> reflateNode(int nodeReference) {
        if (leafNodes.isLeaf(nodeReference)) {
            return new BoundingBox(getLeafPoint(nodeReference));
        }
        if (cachedBoxes[nodeReference] == null) {
            cachedBoxes[nodeReference] = reflateNode(internalNodes.getLeftIndex(nodeReference))
                    .getMergedBox(reflateNode(internalNodes.getRightIndex(nodeReference)));
        }
        return cachedBoxes[nodeReference];
    }

}
