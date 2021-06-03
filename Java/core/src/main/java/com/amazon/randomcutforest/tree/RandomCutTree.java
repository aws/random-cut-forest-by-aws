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
import java.util.Random;

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
public class RandomCutTree extends AbstractRandomCutTree<double[], Node, double[]> {

    public static final int DEFAULT_OUTPUT_AFTER = 64;

    /**
     * @return a new RandomCutTree builder.
     */
    public static Builder<?> builder() {
        return new Builder<>();
    }

    /**
     * Create a new RandomCutTree with optional arguments set to default values.
     *
     * @param randomSeed The random seed used to create the random number generator
     *                   for this tree.
     * @return a new RandomCutTree with optional arguments set to default values.
     */
    public static RandomCutTree defaultTree(long randomSeed) {
        return builder().randomSeed(randomSeed).build();
    }

    /**
     * @return a new RandomCutTree with optional arguments set to default values.
     */
    public static RandomCutTree defaultTree() {
        return builder().random(new Random()).build();
    }

    /**
     * @return true if nodes in this tree retain the center of mass, false
     *         otherwise.
     */
    public boolean centerOfMassEnabled() {
        return super.centerOfMassEnabled;
    }

    /**
     * @return true if points in this tree are saved with sequence indexes, false
     *         otherwise.
     */
    public boolean storeSequenceIndexesEnabled() {
        return super.storeSequenceIndexesEnabled;
    }

    protected RandomCutTree(Builder<?> builder) {
        super(builder);
        super.root = null;
        outputAfter = builder.outputAfter.orElse(DEFAULT_OUTPUT_AFTER);
    }

    @Override
    protected boolean leftOf(double[] point, int cutDimension, double cutValue) {
        return point[cutDimension] <= cutValue;
    }

    @Override
    protected boolean equals(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean referenceEquals(double[] oldPointRef, double[] pointRef) {
        return equals(oldPointRef, pointRef);
    }

    @Override
    protected String toString(double[] doubles) {
        return Arrays.toString(doubles);
    }

    @Override
    void setCachedBox(Node node, AbstractBoundingBox<double[]> savedBox) {
        if (cacheRandom.nextDouble() < boundingBoxCacheFraction) {
            node.setBoundingBox((BoundingBox) savedBox);
        }
    }

    @Override
    void addToBox(Node node, double[] point) {
        node.addToBox(point);
    }

    @Override
    void recomputePointSum(Node node) {
        node.recomputePointSum();
    }

    @Override
    protected double[] getPoint(Node node) {
        return node.getLeafPoint();
    }

    @Override
    protected INode<Node> getNode(Node node) {
        return node;
    }

    @Override
    protected AbstractBoundingBox<double[]> getInternalTwoPointBox(double[] first, double[] second) {
        return new BoundingBox(first, second);
    }

    BoundingBox constructBoxInPlace(BoundingBox currentBox, Node nodeReference) {
        return nodeReference.constructBoxInPlace(currentBox);
    }

    @Override
    BoundingBox recomputeBox(Node nodeReference) {
        return nodeReference.recomputeBox();
    }

    @Override
    protected BoundingBox getLeafBoxFromLeafNode(Node node) {
        return new BoundingBox(node.getLeafPoint());
    }

    @Override
    protected BoundingBox getMutableLeafBoxFromLeafNode(Node node) {
        double[] leafPoint = node.getLeafPoint();
        return new BoundingBox(Arrays.copyOf(leafPoint, leafPoint.length), Arrays.copyOf(leafPoint, leafPoint.length),
                0);
    }

    @Override
    protected BoundingBox constructBoxInPlace(Node node) {
        return node.constructBoxInPlace();
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
    protected boolean isLeaf(Node node) {
        return node.isLeaf();
    }

    @Override
    protected int decrementMass(Node node) {
        return node.decrementMass();
    }

    @Override
    protected int incrementMass(Node node) {
        return node.incrementMass();
    }

    @Override
    protected Node getSibling(Node node) {
        Node parent = node.getParent();
        if (parent.getLeftChild() == node) {
            return parent.getRightChild();
        } else if (parent.getRightChild() == node) {
            return parent.getLeftChild();
        }
        throw new IllegalStateException(" incorrect state ");
    }

    @Override
    protected Node getParent(Node node) {
        return node.getParent();
    }

    @Override
    protected void setParent(Node node, Node parent) {
        node.setParent(parent);
    }

    @Override
    protected void delete(Node node) {
    }

    @Override
    protected int getCutDimension(Node node) {
        return node.getCutDimension();
    }

    @Override
    protected double getCutValue(Node node) {
        return node.getCut().getValue();
    }

    @Override
    protected Node getLeftChild(Node node) {
        return node.getLeftChild();
    }

    @Override
    protected Node getRightChild(Node node) {
        return node.getRightChild();
    }

    @Override
    void replaceChild(Node parent, Node child, Node otherNode) {
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
    protected void replaceNodeBySibling(Node grandParent, Node parent, Node node) {
        replaceChild(grandParent, parent, getSibling(node));
    }

    @Override
    protected Node addLeaf(double[] pointIndex) {
        Node candidate = new Node(pointIndex);
        candidate.setMass(1);
        return candidate;
    }

    @Override
    protected Node addNode(Node leftChild, Node rightChild, int cutDimension, double cutValue, int mass) {
        Node candidate = new Node(leftChild, rightChild, new Cut(cutDimension, cutValue), null, centerOfMassEnabled);
        candidate.setMass(mass);
        return candidate;
    }

    @Override
    protected void increaseMassOfAncestors(Node mergedNode) {
        Node parent = mergedNode.getParent();
        while (parent != null) {
            parent.incrementMass();
            parent = parent.getParent();
        }
    }

    @Override
    protected void decreaseMassOfAncestors(Node mergedNode) {
        Node parent = mergedNode.getParent();
        while (parent != null) {
            parent.decrementMass();
            parent = parent.getParent();
        }
    }

    @Override
    void setLeafPointReference(Node node, double[] point) {
    }

    @Override
    protected int getMass(Node node) {
        return node.getMass();
    }

    @Override
    protected void addSequenceIndex(Node node, long uniqueSequenceNumber) {
        node.addSequenceIndex(uniqueSequenceNumber);
    }

    @Override
    protected void deleteSequenceIndex(Node node, long uniqueSequenceNumber) {
        if (storeSequenceIndexesEnabled) {
            if (!node.getSequenceIndexes().contains(uniqueSequenceNumber)) {
                throw new IllegalStateException("Error in sequence index. Inconsistency in trees in delete step.");
            }
        }
        node.deleteSequenceIndex(uniqueSequenceNumber);
    }

    public static class Builder<T extends Builder<T>> extends AbstractRandomCutTree.Builder<T> {
        public RandomCutTree build() {
            return new RandomCutTree(this);
        }
    }
}
