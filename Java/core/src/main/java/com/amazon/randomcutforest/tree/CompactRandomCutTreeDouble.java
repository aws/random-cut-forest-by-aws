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

public class CompactRandomCutTreeDouble extends AbstractCompactRandomCutTree<double[]> {

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore, boolean cacheEnabled,
            boolean centerOfMassEnabled, boolean enableSequenceIndices) {
        super(maxSize, seed, cacheEnabled, centerOfMassEnabled, enableSequenceIndices);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBox[maxSize - 1];
        }
        if (centerOfMassEnabled) {
            pointSum = new double[maxSize - 1][];
        }
    }

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore,
            CompactNodeManager nodeManager, int rootIndex, boolean cacheEnabled) {
        super(maxSize, seed, nodeManager, rootIndex, cacheEnabled);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBox[maxSize - 1];
        }
    }

    @Override
    protected String toString(double[] point) {
        return Arrays.toString(point);
    }

    @Override
    AbstractBoundingBox<double[]> getInternalTwoPointBox(double[] first, double[] second) {
        return new BoundingBox(first, second);
    }

    @Override
    protected boolean checkEqual(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    AbstractBoundingBox<double[]> getLeafBoxFromLeafNode(Integer nodeReference) {
        return new BoundingBox(pointStore.get(nodeManager.getPointIndex(nodeReference)));
    }

    @Override
    void updateDeletePointSum(int nodeRef, double[] point) {
        if (pointSum[nodeRef] == null) {
            pointSum[nodeRef] = new double[point.length];
        }
        for (int i = 0; i < point.length; i++) {
            pointSum[nodeRef][i] += point[i];
        }
    }

    double[] getPointSum(int ref) {
        return nodeManager.isLeaf(ref) ? getPointFromLeafNode(ref) : pointSum[ref];
    }

    @Override
    void updateAddPointSum(Integer mergedNode, double[] point) {
        if (pointSum[mergedNode] == null) {
            pointSum[mergedNode] = new double[point.length];
        }
        double[] leftSum = getPointSum(nodeManager.getLeftChild(mergedNode));
        double[] rightSum = getPointSum(nodeManager.getRightChild(mergedNode));
        for (int i = 0; i < point.length; i++) {
            pointSum[mergedNode][i] = leftSum[i] + rightSum[i];
        }
        int tempNode = mergedNode;
        while (nodeManager.getParent(tempNode) != NULL) {
            tempNode = nodeManager.getParent(tempNode);
            for (int i = 0; i < point.length; i++) {
                pointSum[tempNode][i] += point[i];
            }
        }
    }

    @Override
    protected boolean leftOf(double[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    // the following is for visitors
    @Override
    protected double[] getPoint(Integer node) {
        double[] internal = pointStore.get(nodeManager.getPointIndex(node));
        return Arrays.copyOf(internal, internal.length);
    }

    @Override
    double[] getPointFromLeafNode(Integer node) {
        return pointStore.get(nodeManager.getPointIndex(node));
    }

    /**
     * creates the bounding box of a node/leaf assuming that caching is enabled
     *
     * @param nodeReference node in question
     * @return the bounding box
     */

    AbstractBoundingBox<double[]> getBoundingBoxReflate(Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return new BoundingBox(getPointFromLeafNode(nodeReference));
        }
        if (cachedBoxes[nodeReference] == null) {
            cachedBoxes[nodeReference] = getBoundingBoxReflate(nodeManager.getLeftChild(nodeReference))
                    .getMergedBox(getBoundingBoxReflate(nodeManager.getRightChild(nodeReference)));
        }
        return cachedBoxes[nodeReference];
    }
}
