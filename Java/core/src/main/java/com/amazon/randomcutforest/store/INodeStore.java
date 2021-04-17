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

package com.amazon.randomcutforest.store;

/**
 * An interface for describing access to the node ntore for different types of
 * trees.
 *
 * his handles leaf nodes which corresponds to [0 .. upperRangeLimit] *
 */
public interface INodeStore {
    /**
     * creates a new internal node
     * 
     * @param parentIndex  parent of the node, can be NULL for root
     * @param leftIndex    left child (cannot be NULL)
     * @param rightIndex   right child (cannot be NULL)
     * @param cutDimension dimension of random cut
     * @param cutValue     value of the random cut
     * @param mass         mass of the subtree at this node
     * @return index of the new node
     */
    int addNode(int parentIndex, int leftIndex, int rightIndex, int cutDimension, double cutValue, int mass);

    /**
     * adds a leaf node
     *
     * @param parentIndex parent of the leaf
     * @param pointIndex  index in point store determining the associated point
     * @param mass        number of copies
     * @return index of the leaf node
     */

    int addLeaf(int parentIndex, int pointIndex, int mass);

    /**
     * gets the index of the point associated with the leaf
     *
     * @param index node
     * @return index of the point in Point Store
     */
    int getPointIndex(int index);

    /**
     * sets/replaces the point index in a leafstore entry
     *
     * @param index      node
     * @param pointIndex the new index of the point corresponding to the leaf
     * @return the older index (useful for audit)
     */

    int setPointIndex(int index, int pointIndex);

    /**
     * set the parent of a node
     * 
     * @param index  node
     * @param parent parent of the node (can be NULL)
     */
    void setParent(int index, int parent);

    /**
     * gets the parent of a node
     * 
     * @param index node
     * @return indef of parent
     */
    int getParent(int index);

    /**
     * deletes a node
     * 
     * @param index node
     */
    void delete(int index);

    /**
     * replaces node oldIndex with provided parent by newIndex, but does not change
     * newIndex because newIndex need not be an internal node.
     * 
     * @param parent   parent of newIndex (cannot be NULL)
     * @param oldIndex node
     * @param newIndex new node to take the same position of oldIndex in the tree
     */

    void replaceChild(int parent, int oldIndex, int newIndex);

    /**
     * gets rightChild
     * 
     * @param index node
     * @return index of rightchild (can be leaf)
     */
    int getRightIndex(int index);

    /**
     * sets rightChild
     *
     * @param index    node
     * @param newIndex new right child
     */
    void setRightIndex(int index, int newIndex);

    /**
     * gets leftChild
     * 
     * @param index node
     * @return index of left child (can be leaf)
     */
    int getLeftIndex(int index);

    /**
     * sets leftChild
     *
     * @param index    node
     * @param newIndex new left child
     */
    void setLeftIndex(int index, int newIndex);

    /**
     * increments mass of node by 1 and returns the new value
     * 
     * @param index node
     * @return new mass
     */

    int incrementMass(int index);

    /**
     * decrements mass of by 1 and returns the new mass; does not delete nodes of
     * zero or any other mass
     * 
     * @param index node
     * @return new mass
     */

    int decrementMass(int index);

    /**
     * gets the cut dimension associated with the (internal) node
     * 
     * @param index node
     * @return dimension of random cut
     */
    int getCutDimension(int index);

    /**
     * gets the cut value associated with the (internal) node
     * 
     * @param index node
     * @return cut value
     */
    double getCutValue(int index);

    /**
     * returns the current mass (number of nodes in he subtree under node)
     * 
     * @param index node
     * @return the mass of the node
     */
    int getMass(int index);

    /**
     * sets the current mass (number of nodes in he subtree under node) useful for
     * rebuilding a tree
     *
     * @param index   node
     * @param newMass the new mass of the node
     */
    void setMass(int index, int newMass);

    /**
     * increases the mass of node as well as all its ancestors by 1 note that all
     * these nodes are internal nodes and we eliminate the back and forth in a
     * single call
     * 
     * @param index node
     */
    void increaseMassOfSelfAndAncestors(int index);

    /**
     * decreases the mass of node as well as all its ancestors by 1 note that all
     * these nodes are internal nodes and we eliminate the back and forth in a
     * single call
     *
     * @param index node
     */
    void decreaseMassOfSelfAndAncestors(int index);

    /**
     * returns the sibling of node, both of whom save the same parent
     * 
     * @param parent the parent
     * @param node   the node
     * @return the sibling
     */
    public int getSibling(int parent, int node);

    boolean isLeaf(int index);

    int getCapacity();

    int computeLeafIndex(int index);

    int size();

}
