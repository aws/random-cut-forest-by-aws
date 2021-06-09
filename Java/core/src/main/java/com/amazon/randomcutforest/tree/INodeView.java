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

import java.util.Set;

public interface INodeView {
    int getMass();

    IBoundingBoxView getBoundingBox();

    IBoundingBoxView getSiblingBoundingBox(double[] point);

    int getCutDimension();

    double getCutValue();

    double[] getLeafPoint();

    default double[] getLiftedLeafPoint() {
        return getLeafPoint();
    };

    Set<Long> getSequenceIndexes();

}
