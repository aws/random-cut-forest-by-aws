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
 * An interface for accessing leafstore for different trees. While the library
 * provides a LeafStore, the interface will allow extensions to other
 * implementations as well as unification with pointer based nodes.
 *
 * * This handles leaf nodes which corresponds to [lowerRangeLimit .. max
 * (short)]
 */
public interface ILeafStore {

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
     * adds parent of the node
     * 
     * @param index  leaf node
     * @param parent parent of the leaf (not a leaf node, can be NULL)
     */

    void setParent(int index, int parent);

    /**
     * returns the parent of the node
     * 
     * @param index node
     * @return index of parent (either NULL or an internal node)
     */
    int getParent(int index);

    /**
     * delets the leaf node
     * 
     * @param index node
     */
    void delete(int index);

    /**
     * gets the index of the point associated with the leaf
     * 
     * @param index node
     * @return index of the point in Point Store
     */
    int getPointIndex(int index);

    /**
     * sets the index of the point associated with the leaf
     *
     * @param index node
     * @return old index in the leafstore so that we can verify (and undo)
     */
    int setPointIndex(int index, int pointIndex);

    /**
     * increases the mass of the leaf and returns the value
     * 
     * @param index node
     * @return current mass (number of copies of the point in the tree)
     */

    int incrementMass(int index);

    /**
     * returns the new mass of the leaf (does not delete)
     * 
     * @param index node
     * @return new mass after decrement by 1
     */

    int decrementMass(int index);

    /**
     * return current number of copies of the point in the tree
     * 
     * @param index node
     * @return current mass
     */
    int getMass(int index);
}
