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

import java.util.HashMap;

public interface INodeView {

    boolean isLeaf();

    int getMass();

    IBoundingBoxView getBoundingBox();

    IBoundingBoxView getSiblingBoundingBox(float[] point);

    int getCutDimension();

    double getCutValue();

    float[] getLeafPoint();

    default float[] getLiftedLeafPoint() {
        return getLeafPoint();
    };

    /**
     * for a leaf node, return the sequence indices corresponding leaf point. If
     * this method is invoked on a non-leaf node then it throws an
     * IllegalStateException.
     */
    HashMap<Long, Integer> getSequenceIndexes();

    /**
     * provides the probability of separation vis-a-vis the bounding box at the node
     * 
     * @param point input piint being evaluated
     * @return the probability of separation
     */

    double probailityOfSeparation(float[] point);

    /**
     * for a leaf node, return the index in the point store for the leaf point. If
     * this method is invoked on a non-leaf node then it throws an
     * IllegalStateException.
     */
    int getLeafPointIndex();

}
