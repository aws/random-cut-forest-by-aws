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

import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;

/**
 * An interface for describing access to the node store for different types of
 * trees.
 */
public class CompactNodeManager {

    private final LeafStore leafstore;
    private final NodeStore nodeStore;
    private final int capacity;
    private final short NULL = -1;

    CompactNodeManager(NodeStore nodeStore, LeafStore leafStore, int capacity) {
        this.leafstore = leafStore;
        this.nodeStore = nodeStore;
        this.capacity = capacity;
    }

    protected int intValue(Integer ref) {
        return ref == null ? NULL : ref.intValue();
    }

    public Integer addNode(Integer parent, Integer leftChild, Integer rightChild, int cutDimension, double cutValue,
            int mass) {
        int parentIndex = intValue(parent);
        int leftIndex = intValue(leftChild);
        int rightIndex = intValue(rightChild);
        return nodeStore.addNode(parentIndex, leftIndex, rightIndex, cutDimension, cutValue, mass);
    }

    public void setParent(Integer child, Integer parent) {
        int parentIndex = intValue(parent);
        if (isLeaf(child)) {
            leafstore.setParent(child, parentIndex);
        } else {
            nodeStore.setParent(child, parentIndex);
        }
    }

    public Integer getParent(Integer child) {
        int val = getInternalParent(child);
        return (val == NULL) ? null : val;
    }

    public boolean parentEquals(Integer child, Integer potentialParent) {
        int val;
        if (isLeaf(child)) {
            val = leafstore.getParent(child);
        } else {
            val = nodeStore.getParent(child);
        }
        if (potentialParent == null) {
            return val == NULL;
        }
        return val == potentialParent.intValue();
    }

    public void delete(Integer node) {
        if (isLeaf(node)) {
            leafstore.delete(node);
        } else {
            nodeStore.delete(node);
        }
    }

    public Integer getRightChild(Integer node) {
        return nodeStore.getRightIndex(node);
    }

    public Integer getLeftChild(Integer node) {
        return nodeStore.getLeftIndex(node);
    }

    public int incrementMass(Integer node) {
        return isLeaf(node) ? leafstore.incrementMass(node) : nodeStore.incrementMass(node);
    }

    public int decrementMass(Integer node) {
        return isLeaf(node) ? leafstore.decrementMass(node) : nodeStore.decrementMass(node);
    }

    public int getCutDimension(Integer node) {
        return nodeStore.getCutDimension(node);
    }

    public double getCutValue(Integer node) {
        return nodeStore.getCutValue(node.intValue());
    }

    public int getMass(Integer node) {
        return isLeaf(node) ? leafstore.getMass(node) : nodeStore.getMass(node);
    }

    public boolean isLeaf(Integer index) {
        checkState(index != null, "illegal");
        return index.intValue() >= capacity;
    }

    public void increaseMassOfAncestorsRecursively(Integer node) {
        int parent = getInternalParent(node);
        nodeStore.increaseMassOfAncestorsAndItselfRecursively(parent);
    }

    public void replaceNode(Integer parent, Integer firstNode, Integer secondNode) {
        nodeStore.replaceNode(parent.intValue(), firstNode.intValue(), secondNode.intValue());
        setParent(secondNode, parent);
    }

    public void replaceParentBySiblingOfNode(Integer grandParent, Integer parent, Integer node) {
        int sibling = nodeStore.getSibling(parent, node);
        nodeStore.replaceNode(grandParent, parent, sibling);
        setParent(sibling, grandParent);
    }

    int getInternalParent(Integer node) {
        Integer boxParent;
        if (isLeaf(node)) {
            boxParent = leafstore.getParent(node);
        } else {
            boxParent = nodeStore.getParent(node);
        }
        return intValue(boxParent);
    }

    public Integer getSibling(Integer node) {
        int parent = getInternalParent(node);
        checkState(parent != NULL, "illegal");
        return nodeStore.getSibling(parent, node.intValue());
    }

    public Integer getPointIndex(Integer index) {
        checkState(isLeaf(index), "illegal state");
        return leafstore.getPointIndex(index);
    }

    public Integer addLeaf(Integer parent, Integer pointIndex, int mass) {
        int parentIndex = intValue(parent);
        return leafstore.addLeaf(parentIndex, pointIndex, mass);
    }

    public NodeStore getNodeStore() {
        return nodeStore;
    }

    public LeafStore getLeafStore() {
        return leafstore;
    }
}
