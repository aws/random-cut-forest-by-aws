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

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore) {
        super(maxSize, seed);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        cachedBoxes = new BoundingBoxFloat[maxSize - 1];
    }

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, LeafStore leafStore,
            NodeStore nodeStore, short rootIndex) {
        super(maxSize, seed, leafStore, nodeStore, rootIndex);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        cachedBoxes = new BoundingBoxFloat[maxSize - 1];
    }

    @Override
    String printString(float[] point) {
        return Arrays.toString(point);
    }

    @Override
    IBox<float[]> getLeafBox(int offset) {
        return new BoundingBoxFloat(pointStore.get(leafNodes.pointIndex[offset]));
    }

    @Override
    protected boolean leftOf(double[] point, short nodeOffset) {
        return point[internalNodes.cutDimension[nodeOffset]] <= internalNodes.cutValue[nodeOffset];
    }

    @Override
    IBox<float[]> getInternalMergedBox(float[] point, float[] oldPoint) {
        return BoundingBoxFloat.getMergedBox(point, oldPoint);
    }

    protected boolean leftOf(float[] point, short nodeOffset) {
        return leftOf(point, internalNodes.cutDimension[nodeOffset], internalNodes.cutValue[nodeOffset]);
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
    protected double[] getLeafPoint(short nodeOffset) {
        return toDoubleArray(pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]));
    }

    BoundingBoxFloat reflateNode(short offset) {
        if (offset - maxSize >= 0) {
            return new BoundingBoxFloat(pointStore.get(leafNodes.pointIndex[offset - maxSize]));
        }
        BoundingBoxFloat newBox = reflateNode(internalNodes.leftIndex[offset])
                .getMergedBox(reflateNode(internalNodes.rightIndex[offset]));
        cachedBoxes[offset] = newBox;
        return newBox;
    }

}
