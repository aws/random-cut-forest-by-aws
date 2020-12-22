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

import java.util.Arrays;

import com.amazon.randomcutforest.Visitor;

/**
 * A Random Cut Tree is a tree data structure whose leaves represent points
 * inserted into the tree and whose interior nodes represent regions of space
 * defined by Bounding Boxes and Cuts. New nodes and leaves are added to the
 * tree by making random cuts.
 *
 * This tree is implemented based on pointers and uses he same logic as the
 * Compact Random Cut trees, providing an alternate implementation to
 * RandomCutTree corresponding to the initial version.
 *
 * The main use of this class is to be updated with points sampled from a
 * stream, and to define traversal methods. Users can then implement a
 * {@link Visitor} which can be submitted to a traversal method in order to
 * compute a statistic from the tree.
 */
public class PointerTree extends AbstractRandomCutTree<double[], Node, double[]> {

    public PointerTree(long seed, boolean enableCache, boolean enableCenterOfMass, boolean enableSequenceIndices) {
        super(seed, enableCache, enableCenterOfMass, enableSequenceIndices);
        rootIndex = null;
    }

    @Override
    protected boolean leftOf(double[] point, int cutDimension, double cutValue) {
        return point[cutDimension] <= cutValue;
    }

    @Override
    boolean checkEqual(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    String toString(double[] doubles) {
        return Arrays.toString(doubles);
    }

    void updateBox(Node nodeReference) {
        checkState(enableCache, " incorrect invocation");
        BoundingBox rightBox = nodeReference.getRightChild().getBoundingBox();
        BoundingBox leftBox = nodeReference.getLeftChild().getBoundingBox();
        nodeReference.setBoundingBox(leftBox.getMergedBox(rightBox));
    }

    @Override
    boolean updateDeletePointBoxes(Node nodeReference, double[] point, boolean isResolved) {
        if (!isResolved && enableCache) {
            updateBox(nodeReference);
        } else {
            nodeReference.setBoundingBox(null);
        }
        return isResolved;
    }

    @Override
    void updateAddPointBoxes(AbstractBoundingBox<double[]> savedBox, Node mergedNode, double[] point,
            Node parentIndex) {

        if (enableCache) {
            Node tempNode = mergedNode;
            while (tempNode != parentIndex) {
                updateBox(tempNode);
                tempNode = tempNode.getParent();
            }
        } else {
            mergedNode.setBoundingBox(null);
        }
    }

    @Override
    protected double[] getPoint(Node node) {
        return new double[0];
    }

    @Override
    INode<Node> getNodeView(Node node) {
        return node;
    }

    @Override
    AbstractBoundingBox<double[]> getInternalTwoPointBox(double[] first, double[] second) {
        return new BoundingBox(first, second);
    }

    @Override
    BoundingBox getBoundingBoxReflate(Node nodeReference) {
        if (isLeaf(nodeReference)) {
            return new BoundingBox(nodeReference.getLeafPoint());
        }
        if (nodeReference.getBoundingBox() == null) {
            nodeReference.setBoundingBox(getBoundingBoxReflate(nodeReference.getLeftChild())
                    .getMergedBox(getBoundingBoxReflate(nodeReference.getRightChild())));
        }
        return nodeReference.getBoundingBox();
    }

    @Override
    AbstractBoundingBox<double[]> getLeafBoxFromLeafNode(Node node) {
        return new BoundingBox(node.getLeafPoint());
    }

    // gets the actual point
    @Override
    double[] getPointFromLeafNode(Node node) {
        checkState(node.isLeaf(), "Incorrect use");
        return node.getLeafPoint();
    }

    @Override
    double[] getPointFromPointReference(double[] pointIndex) {
        return pointIndex;
    }

    // gets the reference, which happens to be the same point
    @Override
    double[] getPointReference(Node node) {
        return node.getLeafPoint();
    }

    @Override
    boolean isLeaf(Node node) {
        return node.isLeaf();
    }

    @Override
    int decrementMass(Node node) {
        return node.decrementMass();
    }

    @Override
    int incrementMass(Node node) {
        return node.incrementMass();
    }

    @Override
    Node getSibling(Node node) {
        Node parent = node.getParent();
        if (parent.getLeftChild() == node) {
            return parent.getRightChild();
        } else if (parent.getRightChild() == node) {
            return parent.getLeftChild();
        }
        throw new IllegalStateException(" incorrect state ");
    }

    @Override
    Node getParent(Node node) {
        return node.getParent();
    }

    @Override
    void setParent(Node node, Node parent) {
        node.setParent(parent);
    }

    @Override
    void delete(Node node) {
    }

    @Override
    int getCutDimension(Node node) {
        return node.getCutDimension();
    }

    @Override
    double getCutValue(Node node) {
        return node.getCut().getValue();
    }

    @Override
    Node getLeftChild(Node node) {
        return node.getLeftChild();
    }

    @Override
    Node getRightChild(Node node) {
        return node.getRightChild();
    }

    @Override
    void replaceNode(Node parent, Node child, Node otherNode) {
        if (parent.getLeftChild() == child) {
            parent.setLeftChild(otherNode);
        } else if (parent.getRightChild() == child) {
            parent.setRightChild(otherNode);
        } else {
            throw new IllegalStateException(" incorrect state ");
        }
        otherNode.setParent(parent);
    }

    @Override
    void replaceNodeBySibling(Node grandParent, Node parent, Node node) {
        replaceNode(grandParent, parent, getSibling(node));
    }

    @Override
    Node addLeaf(Node parent, double[] pointIndex, int mass) {
        Node candidate = new Node(pointIndex);
        candidate.setMass(mass);
        candidate.setParent(parent);
        return candidate;
    }

    @Override
    Node addNode(Node parent, Node leftChild, Node rightChild, int cutDimension, double cutValue, int mass) {
        Node candidate = new Node(leftChild, rightChild, new Cut(cutDimension, cutValue), null, false);
        candidate.setParent(parent);
        candidate.setMass(mass);
        return candidate;
    }

    @Override
    void increaseMassOfAncestorsRecursively(Node mergedNode) {
        Node parent = mergedNode.getParent();
        while (parent != null) {
            parent.incrementMass();
            parent = parent.getParent();
        }
    }

    @Override
    int getMass(Node node) {
        return node.getMass();
    }

    @Override
    void addSequences(Node node, long uniqueSequenceNumber) {
        node.addSequenceIndex(uniqueSequenceNumber);
    }

    @Override
    void deleteSequences(Node node, long uniqueSequenceNumber) {
        node.deleteSequenceIndex(uniqueSequenceNumber);
    }

}
