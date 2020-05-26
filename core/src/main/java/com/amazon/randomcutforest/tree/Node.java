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
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import com.amazon.randomcutforest.Visitor;

/**
 * A Node in a {@link RandomCutTree}. All nodes contain references to the parent
 * and children nodes (which may be null, in the case of the root node or a leaf
 * node). All nodes also contain a {@link BoundingBox}, which is the smallest
 * BoundingBox that contains all points that are descendents of the given node.
 * Leaf nodes additionally contain a point value.
 */
public class Node {

    /**
     * For a leaf node this contains the leaf point, for a non-leaf node this value
     * will be null.
     */
    private final double[] leafPoint;
    /**
     * If this is a non-leaf node and the center of mass computation has been
     * enabled in RandomCutTree, this array will store the sum of all descendent
     * points.
     */
    private final double[] pointSum;
    /**
     * Parent of this node.
     */
    private Node parent;
    /**
     * Right child of this node.
     */
    private Node rightChild;
    /**
     * Left child of this node.
     */
    private Node leftChild;
    /**
     * For a non-leaf node this is a {@link Cut} that divides this node's bounding
     * box into two sections. This cut also determines the canonical root-to-leaf
     * traversal path for a given point.
     *
     * @see RandomCutTree#traverseTree(double[], Visitor)
     */
    private Cut cut;
    /**
     * The smallest {@link BoundingBox} that contains all descendent points for this
     * node.
     */
    private BoundingBox boundingBox;
    /**
     * The number of descendent points for this node.
     */
    private int mass;
    /**
     * If this is a leaf node and sequence indexes have been enabled in the
     * RandomCutTree, this set stores the indexes corresponding to times when the
     * given leaf point was added to the tree.
     */
    private Set<Long> sequenceIndexes;

    /**
     * Create a new non-leaf Node.
     *
     * @param leftChild          Left child of the new node.
     * @param rightChild         Right child of the new node.
     * @param cut                A Cut that divides the bounding box into two
     *                           sections.
     * @param boundingBox        The bounding box for this node.
     * @param enableCenterOfMass A flag indicating whether center of mass
     *                           computations will be applied tot his node.
     */
    public Node(final Node leftChild, final Node rightChild, final Cut cut, final BoundingBox boundingBox,
            boolean enableCenterOfMass) {
        this.rightChild = rightChild;
        this.leftChild = leftChild;
        this.cut = cut;
        this.boundingBox = boundingBox;
        this.sequenceIndexes = null;
        if (!enableCenterOfMass) {
            this.pointSum = null;
        } else {
            this.pointSum = new double[boundingBox.getDimensions()];
        }
        leafPoint = null;
    }

    /**
     * Create a new non-leaf Node with center of mass computation disabled.
     *
     * @param leftChild   Left child of the new node.
     * @param rightChild  Right child of the new node.
     * @param cut         A Cut that divides the bounding box into two sections.
     * @param boundingBox The bounding box for this node.
     */
    public Node(final Node leftChild, final Node rightChild, final Cut cut, final BoundingBox boundingBox) {
        this(leftChild, rightChild, cut, boundingBox, false);
    }

    /**
     * Create a new leaf node.
     *
     * @param leafPoint The point that defines the new leaf node.
     */
    public Node(double[] leafPoint) {
        this.leafPoint = leafPoint;
        this.sequenceIndexes = null;
        this.pointSum = null;
        boundingBox = new BoundingBox(leafPoint);
    }

    /**
     * Test if the point is "left" of the {@link Cut} in the given Node.
     * 
     * @param point A point that we are testing in comparison to the given node
     * @param node  A non-leaf node
     * @return true if the point is left of the cut in the given node, false
     *         otherwise.
     *
     * @see Cut#isLeftOf(double[], Cut)
     */
    public static boolean isLeftOf(double[] point, Node node) {
        return Cut.isLeftOf(point, node.cut);
    }

    /**
     * @return the parent node of this node.
     */
    public Node getParent() {
        return parent;
    }

    /**
     * Set the parent node to the given value.
     * 
     * @param parent The new parent node.
     */
    protected void setParent(final Node parent) {
        this.parent = parent;
    }

    /**
     * @return the right child of this node.
     */
    public Node getRightChild() {
        return rightChild;
    }

    /**
     * Set the right child to the given value.
     * 
     * @param rightChild The new right child.
     * @throws IllegalStateException if this is a leaf node.
     */
    protected void setRightChild(final Node rightChild) {
        checkState(!isLeaf(), "Cannot assign child to a leaf node");
        this.rightChild = rightChild;
    }

    /**
     * @return the left child of this node.
     */
    public Node getLeftChild() {
        return leftChild;
    }

    /**
     * Set the left child to the given value.
     * 
     * @param leftChild The new left child.
     * @throws IllegalStateException if this is a leaf node.
     */
    protected void setLeftChild(final Node leftChild) {
        checkState(!isLeaf(), "Cannot assign child to a leaf node");
        this.leftChild = leftChild;
    }

    /**
     * Return this node's bounding box. This will be the smallest bounding box that
     * contains all descendent points for this node.
     * 
     * @return this node's bounding box.
     */
    public BoundingBox getBoundingBox() {
        return boundingBox;
    }

    /**
     * Set the node's bounding box to the given value.
     * 
     * @param boundingBox The new bounding box.
     */
    protected void setBoundingBox(final BoundingBox boundingBox) {
        this.boundingBox = boundingBox;
    }

    /**
     * For a leaf node, test whether this given point is equal to the leaf point.
     * This test equality as double values (that is, the method will not return true
     * of the given point and the leaf point are merely close). For a non-leaf node
     * this method will always return false.
     *
     * @param point A point that we are comparing to this node's leaf point.
     * @return true if this is a leaf node and the point is equal to the leaf point,
     *         false otherwise.
     */
    public boolean leafPointEquals(double[] point) {
        return Arrays.equals(leafPoint, point);
    }

    /**
     * For a leaf node, return a copy of the leaf point.
     * 
     * @return a copy of the leaf point.
     * @throws IllegalStateException if this is not a leaf node.
     */
    public double[] getLeafPoint() {
        checkState(isLeaf(), "Not a leaf node");
        return Arrays.copyOf(leafPoint, leafPoint.length);
    }

    /**
     * For a leaf node, return the value of the leaf point at the ith coordinate.
     * 
     * @param i A coordinate value.
     * @return the value of the leaf point at the ith coordinage.
     * @throws IllegalStateException if this is not a leaf node.
     */
    public double getLeafPoint(int i) {
        checkState(isLeaf(), "Not a leaf node");
        return leafPoint[i];
    }

    /**
     * Return the mass of this node. For a leaf node, this will be equal to the
     * number of times that the leaf point has been added to the tree. For a
     * non-leaf node, this will be equal to the total mass of all descendent points.
     * 
     * @return the mass of this node.
     */
    public int getMass() {
        return mass;
    }

    /**
     * Set this node's mass to the given value.
     * 
     * @param mass The new mass value.
     */
    protected void setMass(int mass) {
        this.mass = mass;
    }

    /**
     * Add the mass delta to the existing mass.
     *
     * @param massDelta change in mass delta
     */
    protected void addMass(int massDelta) {
        mass += massDelta;
    }

    /**
     * Increment this node's mass by 1.
     */
    protected void incrementMass() {
        addMass(1);
    }

    /**
     * Decrement this node's mass by 1.
     */
    protected void decrementMass() {
        addMass(-1);
    }

    /**
     * @return this node's cut value for a non-leaf node, null otherwise.
     */
    public Cut getCut() {
        return cut;
    }

    /**
     * @return true if the node is a leaf node, false otherwise.
     */
    public boolean isLeaf() {
        return leafPoint != null;
    }

    /**
     * Subtract the given point from this node's point sum.
     * 
     * @param point A point value to subtract from this node's point sum.
     */
    protected void subtractFromPointSum(double[] point) {
        checkState(pointSum != null, "center of mass computation is disabled");
        for (int i = 0; i < point.length; i++) {
            pointSum[i] -= point[i];
        }
    }

    /**
     * Add the given point to this node's point sum.
     * 
     * @param point A point value to add to this node's point sum.
     */
    protected void addToPointSum(double[] point) {
        checkState(pointSum != null, "center of mass computation is disabled");
        for (int i = 0; i < point.length; i++) {
            pointSum[i] += point[i];
        }
    }

    /**
     * Return the value of this node's point sum. For leaf node, this will be the
     * leaf mass times the leaf point value. For a non-leaf node with center of mass
     * computation enabled, this will be the sum of all descendent points. For a
     * non-leaf node with center of mass computation disabled, this will be an array
     * of 0s.
     * 
     * @return the value of this node's point sum.
     */
    public double[] getPointSum() {
        double[] result = new double[boundingBox.getDimensions()];
        // makes a new copy to avoid altering the sum
        if (leafPoint != null) {
            for (int i = 0; i < boundingBox.getDimensions(); i++) {
                result[i] = mass * leafPoint[i];
            }
        } else {
            if (pointSum != null) {
                if (boundingBox.getDimensions() >= 0) {
                    System.arraycopy(pointSum, 0, result, 0, pointSum.length);
                }
            }
        }
        return result;
    }

    /**
     * If the option to compute center of mass is enabled in the RandomCutTree that
     * this Node belongs to, this method will return the center of mass of all the
     * points contained in this node. If the option is not enabled, it returns an
     * array of all 0s.
     *
     * @return the center of mass of this node or the zero array if the option is
     *         disabled.
     */
    public double[] getCenterOfMass() {
        // this will be 0 if the corresponding flag is not set in the forest
        double[] result = new double[boundingBox.getDimensions()];
        // makes a new copy to avoid altering the sum
        if (leafPoint != null) {
            System.arraycopy(leafPoint, 0, result, 0, boundingBox.getDimensions());
        } else {
            if (pointSum != null) {
                for (int i = 0; i < boundingBox.getDimensions(); i++) {
                    result[i] = pointSum[i] / mass;
                }
            }
        }
        return result;
    }

    /**
     * Return an unmodifiable set of sequence indexes. These are ordinals which
     * indicate the times when this point was added to the tree. If this is an
     * interior node or if storing sequence indexes is disable, the set will be
     * empty.
     *
     * @return an unmodifiable set of sequence indexes.
     */
    public Set<Long> getSequenceIndexes() {
        if (sequenceIndexes != null) {
            return Collections.unmodifiableSet(sequenceIndexes);
        } else {
            return Collections.emptySet();
        }
    }

    /**
     * Add a new sequence index to this node.
     * 
     * @param sequenceIndex A new sequence index to be added to this node..
     */
    protected void addSequenceIndex(long sequenceIndex) {
        if (sequenceIndexes == null) {
            sequenceIndexes = new HashSet<>();
        }
        sequenceIndexes.add(sequenceIndex);
    }

    /**
     * Remove the given sequence index from this node.
     * 
     * @param sequenceIndex A sequence index to remove.
     */
    protected void deleteSequenceIndex(long sequenceIndex) {
        if (sequenceIndexes != null) {
            sequenceIndexes.remove(sequenceIndex);
        }
    }
}
