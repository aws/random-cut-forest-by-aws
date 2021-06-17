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
import java.util.Random;

import com.amazon.randomcutforest.store.IPointStoreView;

public class CompactRandomCutTreeFloat extends AbstractCompactRandomCutTree<float[]> {

    public static CompactRandomCutTreeFloat.Builder builder() {
        return new Builder();
    }

    protected CompactRandomCutTreeFloat(CompactRandomCutTreeFloat.Builder builder) {
        super(builder);
        checkNotNull(builder.pointStoreView, "pointStore must not be null");
        super.pointStore = builder.pointStoreView;
        super.boxCache = new BoxCacheFloat(0L, boundingBoxCacheFraction, maxSize - 1);
        if (builder.centerOfMassEnabled) {
            pointSum = new float[maxSize - 1][];
        }
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
    protected AbstractBoundingBox<float[]> getInternalTwoPointBox(Integer firstRef, Integer secondRef) {
        float[] first = getPointFromPointReference(firstRef);
        float[] second = getPointFromPointReference(secondRef);
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

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box. The cut is over single precision,
     * although the representation is over doubles still.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    @Override
    protected Cut randomCut(Random random, AbstractBoundingBox<?> box) {
        double rangeSum = box.getRangeSum();
        checkArgument(rangeSum > 0, "box.getRangeSum() must be greater than 0");

        double breakPoint = random.nextDouble() * rangeSum;

        for (int i = 0; i < box.getDimensions(); i++) {
            double range = box.getRange(i);
            if (breakPoint <= range) {
                float cutValue = (float) (box.getMinValue(i) + breakPoint);

                // Random cuts have to take a value in the half-open interval [minValue,
                // maxValue) to ensure that a
                // Node has a valid left child and right child.
                if ((cutValue == box.getMaxValue(i)) && (box.getMinValue(i) < box.getMaxValue(i))) {
                    cutValue = Math.nextAfter(cutValue, box.getMinValue(i));
                }

                if (cutValue < (float) box.getMinValue(i)) {
                    throw new IllegalStateException(" precision error in single precision cuts");
                }
                // the cut still stores a double value; but the previous section validates that
                // the
                // cut is meaningful in single precision
                return new Cut(i, cutValue);
            }
            breakPoint -= range;
        }

        throw new IllegalStateException("The break point did not lie inside the expected range");
    }
}
