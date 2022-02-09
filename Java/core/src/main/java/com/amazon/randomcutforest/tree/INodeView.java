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

    HashMap<Long, Integer> getSequenceIndexes();

    double probailityOfSeparation(float[] point);

    int getLeafPointIndex();

}
