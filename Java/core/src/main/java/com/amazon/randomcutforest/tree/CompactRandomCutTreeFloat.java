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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;

import java.util.Arrays;

import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.IPointStoreView;

public class CompactRandomCutTreeFloat extends AbstractCompactRandomCutTree<float[]> {

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, boolean cacheEnabled,
            boolean centerOfMassEnabled, boolean enableSequenceIndices) {
        this(new Builder().pointStore(pointStore).maxSize(maxSize).randomSeed(seed)
                .storeSequenceIndexesEnabled(enableSequenceIndices).centerOfMassEnabled(centerOfMassEnabled)
                .enableBoundingBoxCaching(cacheEnabled));
    }

    public CompactRandomCutTreeFloat(CompactRandomCutTreeFloat.Builder builder) {
        super(builder);
        checkNotNull(builder.pointStoreView, "pointStore must not be null");
        super.pointStore = builder.pointStoreView;
        if (builder.boundingBoxCachingEnabled) {
            cachedBoxes = new BoundingBoxFloat[maxSize - 1];
        }
        if (builder.centerOfMassEnabled) {
            pointSum = new float[maxSize - 1][];
        }
    }

    @Override
    public void swapCaches(int[] map) {
        checkArgument(enableCache, "incorrect call to swapping caches");
        BoundingBoxFloat[] newCache = new BoundingBoxFloat[maxSize - 1];
        for (int i = 0; i < maxSize - 1; i++) {
            if (map[i] != NULL) {
                newCache[map[i]] = (BoundingBoxFloat) cachedBoxes[i];
            }
        }
        cachedBoxes = newCache;
    }

    @Override
    protected String toString(float[] point) {
        return Arrays.toString(point);
    }

    @Override
    protected AbstractBoundingBox<float[]> getLeafBoxFromLeafNode(Integer pointIndex) {
        return new BoundingBoxFloat(pointStore.get(getPointReference(pointIndex)));
    }

    @Override
    protected AbstractBoundingBox<float[]> getMutableLeafBoxFromLeafNode(Integer nodeReference) {
        // pointStore makes an explicit copy
        float[] leafPoint = pointStore.get(getPointReference(nodeReference));
        return new BoundingBoxFloat(leafPoint, Arrays.copyOf(leafPoint, leafPoint.length), 0);
    }

    @Override
    protected AbstractBoundingBox<float[]> getInternalTwoPointBox(float[] first, float[] second) {
        return new BoundingBoxFloat(first, second);
    }

    @Override
    protected boolean equals(float[] oldPoint, float[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean leftOf(float[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    @Override
    protected double[] getPoint(Integer nodeOffset) {
        return toDoubleArray(getPointFromLeafNode(nodeOffset));
    }

    @Override
    void recomputePointSum(Integer node) {
        if (pointSum[node] == null) {
            pointSum[node] = new float[pointStore.getDimensions()];
        }
        float[] leftSum = getPointSum(getLeftChild(node));
        float[] rightSum = getPointSum(getRightChild(node));
        for (int i = 0; i < pointStore.getDimensions(); i++) {
            pointSum[node][i] = leftSum[i] + rightSum[i];
        }
    }

    public static class Builder extends AbstractCompactRandomCutTree.Builder<Builder> {
        private IPointStoreView<float[]> pointStoreView;

        public CompactRandomCutTreeFloat.Builder pointStore(IPointStoreView<float[]> pointStoreView) {
            this.pointStoreView = pointStoreView;
            return this;
        }

        public CompactRandomCutTreeFloat build() {
            return new CompactRandomCutTreeFloat(this);
        }
    }
}
