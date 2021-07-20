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

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.store.IPointStoreView;

public class CompactRandomCutTreeDouble extends AbstractCompactRandomCutTree<double[]> {

    public static CompactRandomCutTreeDouble.Builder builder() {
        return new Builder();
    }

    protected CompactRandomCutTreeDouble(CompactRandomCutTreeDouble.Builder builder) {
        super(builder);
        checkNotNull(builder.pointStoreView, "pointStore must not be null");
        super.pointStore = builder.pointStoreView;
        super.boxCache = new BoxCacheDouble(0L, boundingBoxCacheFraction, maxSize - 1);
        if (builder.centerOfMassEnabled) {
            pointSum = new double[maxSize - 1][];
        }
    }

    @Override
    protected String toString(double[] point) {
        return Arrays.toString(point);
    }

    @Override
    protected AbstractBoundingBox<double[]> getInternalTwoPointBox(Integer firstRef, Integer secondRef) {
        double[] first = getPointFromPointReference(firstRef);
        double[] second = getPointFromPointReference(secondRef);
        return new BoundingBox(first, second);
    }

    @Override
    protected boolean equals(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected AbstractBoundingBox<double[]> getLeafBoxFromLeafNode(Integer nodeReference) {
        return new BoundingBox(pointStore.get(getPointReference(nodeReference)));
    }

    @Override
    protected AbstractBoundingBox<double[]> getMutableLeafBoxFromLeafNode(Integer nodeReference) {
        // pointStore makes an explicit copy
        double[] leafPoint = pointStore.get(getPointReference(nodeReference));
        return new BoundingBox(leafPoint, Arrays.copyOf(leafPoint, leafPoint.length), 0);
    }

    @Override
    protected boolean leftOf(double[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    // the following is for visitors; pointStore makes an explicit copy
    @Override
    protected double[] getPoint(Integer node) {
        return pointStore.get(getPointReference(node));
    }

    @Override
    void recomputePointSum(Integer node) {
        if (pointSum[node] == null) {
            pointSum[node] = new double[pointStore.getDimensions()];
        }
        double[] leftSum = getPointSum(getLeftChild(node));
        double[] rightSum = getPointSum(getRightChild(node));
        for (int i = 0; i < pointStore.getDimensions(); i++) {
            pointSum[node][i] = leftSum[i] + rightSum[i];
        }
    }

    public static class Builder extends AbstractCompactRandomCutTree.Builder<Builder> {
        private IPointStoreView<double[]> pointStoreView;

        public Builder pointStore(IPointStoreView<double[]> pointStoreView) {
            this.pointStoreView = pointStoreView;
            return this;
        }

        public CompactRandomCutTreeDouble build() {
            if (pointStoreView == null) {
                throw new IllegalArgumentException("pointstore cannot be null for compact trees");
            }
            precision = Precision.FLOAT_64;
            return new CompactRandomCutTreeDouble(this);
        }
    }
}
