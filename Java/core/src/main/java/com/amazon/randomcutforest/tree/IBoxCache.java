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

public interface IBoxCache<Point> {

    /**
     * a function that decides if the bounding for an internal node is managed by
     * the cache or otherwise if the bounding box is not managed then it would not
     * be stored and would be rebuilt on demand
     * 
     * @param index internal node
     * @return true if the box is managed and is/can be cached
     */
    boolean containsKey(int index);

    /**
     * sets a box in the cache; the check whether the box is managed by the cache is
     * performed internally
     * 
     * @param index internal node
     * @param box   bounding box corresponding to that node
     */
    void setBox(int index, AbstractBoundingBox<Point> box);

    /**
     *
     * @param index internal node
     * @return the bounding of the node (if present) or null otherwise (even if the
     *         box is not managed)
     */
    AbstractBoundingBox<Point> getBox(int index);

    /**
     * swaps the managed boxes
     * 
     * @param map a renaming of the internal nodes, node i is renamed to map[i]
     */
    void swapCaches(int[] map);

    /**
     * adds a point to a bounding box, if it is managed
     * 
     * @param index internal node
     * @param point point to be added
     */
    void addToBox(int index, Point point);

    /**
     * changes the fraction of boxes cached dynamically
     * 
     * @param fraction new fraction of boxes to be cached
     */

    void setCacheFraction(double fraction);

}
