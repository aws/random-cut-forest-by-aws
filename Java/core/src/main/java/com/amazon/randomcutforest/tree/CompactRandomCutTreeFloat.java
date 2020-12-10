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
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;

import java.util.Arrays;

import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;

public class CompactRandomCutTreeFloat extends AbstractCompactRandomCutTree<float[]> {

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, boolean cacheEnabled) {
        super(maxSize, seed);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBoxFloat[maxSize - 1];
        }
    }

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, LeafStore leafStore,
            NodeStore nodeStore, int rootIndex) {
        super(maxSize, seed, leafStore, nodeStore, rootIndex);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        cachedBoxes = new BoundingBoxFloat[maxSize - 1];
    }

    @Override
    String toString(float[] point) {
        return Arrays.toString(point);
    }

    @Override
    IBoundingBox<float[]> getLeafBox(int pointOffset) {
        return new BoundingBoxFloat(pointStore.get(pointOffset));
    }

    @Override
    IBoundingBox<float[]> getInternalMergedBox(float[] point, float[] oldPoint) {
        return BoundingBoxFloat.getMergedBox(point, oldPoint);
    }

    @Override
    boolean checkEqual(float[] oldPoint, float[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean leftOf(float[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    @Override
    protected double[] getLeafPoint(int nodeOffset) {
        return toDoubleArray(pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]));
    }

    @Override
    BoundingBoxFloat reflateNode(int offset) {
        if (leafNodes.isLeaf(offset)) {
            return new BoundingBoxFloat(pointStore.get(leafNodes.getPointIndex(offset)));
        }
        BoundingBoxFloat newBox = reflateNode(internalNodes.getLeftIndex(offset))
                .getMergedBox(reflateNode(internalNodes.getRightIndex(offset)));
        if (cachedBoxes != null) {
            cachedBoxes[offset] = newBox;
        }
        return newBox;
    }

}
