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

    /**
     * By default, trees will not store sequence indexes.
     */
    public static final boolean DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED = false;

    /**
     * By default, nodes will not store center of mass.
     */
    public static final boolean DEFAULT_CENTER_OF_MASS_ENABLED = false;

    protected RandomCutTree(Builder<?> builder) {
        super(builder.random, builder.boundingBoxCachingEnabled, builder.centerOfMassEnabled,
                builder.storeSequenceIndexesEnabled);
        super.rootIndex = null;
    }

    /**
     * @return a new RandomCutTree builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a new RandomCutTree with optional arguments set to default values.
     *
     * @param randomSeed The random seed used to create the random number generator
     *                   for this tree.
     * @return a new RandomCutTree with optional arguments set to default values.
     */
    public static RandomCutTree defaultTree(long randomSeed) {
        return builder().random(new Random(randomSeed)).build();
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
        return super.enableCenterOfMass;
    }

    /**
     * @return true if points in this tree are saved with sequence indexes, false
     *         otherwise.
     */
    public boolean storeSequenceIndexesEnabled() {
        return super.enableSequenceIndices;
    }

    public RandomCutTree(long seed, boolean enableCache, boolean enableCenterOfMass, boolean enableSequenceIndices) {
        super(seed, enableCache, enableCenterOfMass, enableSequenceIndices);
        rootIndex = null;
    }

    public RandomCutTree(Random random, boolean enableCache, boolean enableCenterOfMass,
            boolean enableSequenceIndices) {
        super(random, enableCache, enableCenterOfMass, enableSequenceIndices);
        rootIndex = null;
    }

    @Override
    protected boolean leftOf(double[] point, int cutDimension, double cutValue) {
        return point[cutDimension] <= cutValue;
    }

    @Override
    protected boolean checkEqual(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected String toString(double[] doubles) {
        return Arrays.toString(doubles);
    }

    @Override
    void setCachedBox(Node node, AbstractBoundingBox<double[]> savedBox) {
        node.setBoundingBox((BoundingBox) savedBox);
    }

    @Override
    void addToBox(Node node, double[] point) {
        node.setBoundingBox(node.getBoundingBox().getMergedBox(point));
    }

    @Override
    void reComputePointSum(Node node, double[] point) {
        node.reComputePointSum(point);
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
    AbstractBoundingBox<double[]> getInternalTwoPointBox(double[] first, double[] second) {
        return new BoundingBox(first, second);
    }

    @Override
    BoundingBox getBoundingBoxReflate(Node nodeReference) {
        if (isLeaf(nodeReference)) {
            return new BoundingBox(nodeReference.getLeafPoint());
        }
        if (nodeReference.getBoundingBox() == null) {
            return recomputeBox(nodeReference);
        }
        return nodeReference.getBoundingBox();
    }

    @Override
    BoundingBox recomputeBox(Node nodeReference) {
        nodeReference.setBoundingBox(getBoundingBoxReflate(nodeReference.getLeftChild())
                .getMergedBox(getBoundingBoxReflate(nodeReference.getRightChild())));
        return nodeReference.getBoundingBox();
    }

    @Override
    AbstractBoundingBox<double[]> getLeafBoxFromLeafNode(Node node) {
        return new BoundingBox(node.getLeafPoint());
    }

    @Override
    AbstractBoundingBox<double[]> getMutableLeafBoxFromLeafNode(Node node) {
        double[] leafPoint = node.getLeafPoint();
        return new BoundingBox(Arrays.copyOf(leafPoint, leafPoint.length), Arrays.copyOf(leafPoint, leafPoint.length),
                0);
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
    protected void replaceNodeBySibling(Node grandParent, Node parent, Node node) {
        replaceNode(grandParent, parent, getSibling(node));
    }

    @Override
    protected Node addLeaf(Node parent, double[] pointIndex, int mass) {
        Node candidate = new Node(pointIndex);
        candidate.setMass(mass);
        candidate.setParent(parent);
        return candidate;
    }

    @Override
    protected Node addNode(Node parent, Node leftChild, Node rightChild, int cutDimension, double cutValue, int mass) {
        Node candidate = new Node(leftChild, rightChild, new Cut(cutDimension, cutValue), null, enableCenterOfMass);
        candidate.setParent(parent);
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
    protected int getMass(Node node) {
        return node.getMass();
    }

    protected Node getRoot() {
        return rootIndex;
    }

    @Override
    protected void addSequenceIndex(Node node, long uniqueSequenceNumber) {
        node.addSequenceIndex(uniqueSequenceNumber);
    }

    @Override
    protected void deleteSequenceIndex(Node node, long uniqueSequenceNumber) {
        if (enableSequenceIndices) {
            if (!node.getSequenceIndexes().contains(uniqueSequenceNumber)) {
                throw new IllegalStateException("Error in sequence index. Inconsistency in trees in delete step.");
            }
        }
        node.deleteSequenceIndex(uniqueSequenceNumber);
    }

    public static class Builder<T extends RandomCutTree.Builder<T>> {
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        private boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        private boolean boundingBoxCachingEnabled = true;
        private Random random;

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public T enableBoundingBoxCaching(boolean boundingBoxCacheEnabled) {
            this.boundingBoxCachingEnabled = boundingBoxCacheEnabled;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.random = new Random(randomSeed);
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
            return (T) this;
        }

        public RandomCutTree build() {
            return new RandomCutTree(random, boundingBoxCachingEnabled, centerOfMassEnabled,
                    storeSequenceIndexesEnabled);
        }
    }

    /**
     * Return the sibling of a non-root node. Note that every non-leaf node in a
     * Random Cut Tree has two children.
     *
     * @param node The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */
    static Node getSiblingStatic(Node node) {
        checkNotNull(node.getParent(), "node parent must not be null");

        Node parent = node.getParent();
        if (parent.getLeftChild() == node) {
            return parent.getRightChild();
        } else if (parent.getRightChild() == node) {
            return parent.getLeftChild();
        } else {
            throw new IllegalArgumentException("node parent does not link back to node");
        }
    }

}
