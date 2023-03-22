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

import java.util.HashMap;

import com.amazon.randomcutforest.store.IPointStoreView;

public class NodeView implements INodeView {

    public static double SWITCH_FRACTION = 0.499;

    RandomCutTree tree;
    int currentNodeOffset;
    float[] leafPoint;
    BoundingBox currentBox;

    public NodeView(RandomCutTree tree, IPointStoreView<float[]> pointStoreView, int root) {
        this.currentNodeOffset = root;
        this.tree = tree;
    }

    public int getMass() {
        return tree.getMass(currentNodeOffset);
    }

    public IBoundingBoxView getBoundingBox() {
        if (currentBox == null) {
            return tree.getBox(currentNodeOffset);
        }
        return currentBox;
    }

    public IBoundingBoxView getSiblingBoundingBox(float[] point) {
        return (toLeft(point)) ? tree.getBox(tree.nodeStore.getRightIndex(currentNodeOffset))
                : tree.getBox(tree.nodeStore.getLeftIndex(currentNodeOffset));
    }

    public int getCutDimension() {
        return tree.nodeStore.getCutDimension(currentNodeOffset);
    }

    @Override
    public double getCutValue() {
        return tree.nodeStore.getCutValue(currentNodeOffset);
    }

    public float[] getLeafPoint() {
        return leafPoint;
    }

    public HashMap<Long, Integer> getSequenceIndexes() {
        checkState(isLeaf(), "can only be invoked for a leaf");
        if (tree.storeSequenceIndexesEnabled) {
            return tree.getSequenceMap(tree.getPointIndex(currentNodeOffset));
        } else {
            return new HashMap<>();
        }
    }

    @Override
    public double probailityOfSeparation(float[] point) {
        return tree.probabilityOfCut(currentNodeOffset, point, currentBox);
    }

    @Override
    public int getLeafPointIndex() {
        return tree.getPointIndex(currentNodeOffset);
    }

    public boolean isLeaf() {
        return tree.nodeStore.isLeaf(currentNodeOffset);
    }

    protected void setCurrentNode(int newNode, int index, boolean setBox) {
        currentNodeOffset = newNode;
        leafPoint = tree.pointStoreView.get(index);
        if (setBox && tree.boundingBoxCacheFraction < SWITCH_FRACTION) {
            currentBox = new BoundingBox(leafPoint, leafPoint);
        }
    }

    protected void setCurrentNodeOnly(int newNode) {
        currentNodeOffset = newNode;
    }

    public void updateToParent(int parent, int currentSibling, boolean updateBox) {
        currentNodeOffset = parent;
        if (updateBox && tree.boundingBoxCacheFraction < SWITCH_FRACTION) {
            tree.growNodeBox(currentBox, tree.pointStoreView, parent, currentSibling);
        }
    }

    // this function exists for matching the behavior of RCF2.0 and will be replaced
    // this function explicitly uses the encoding of the new nodestore
    protected boolean toLeft(float[] point) {
        return point[tree.nodeStore.getCutDimension(currentNodeOffset)] <= tree.nodeStore
                .getCutValue(currentNodeOffset);
    }
}
