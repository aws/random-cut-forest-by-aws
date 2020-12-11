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

public interface IBoundingBox<P> {

    double getRangeSum();

    int getDimensions();

    double getRange(int i);

    double getMinValue(int i);

    double getMaxValue(int i);

    boolean contains(P point);

    // duplicates
    IBoundingBox<P> copy();

    // below keeps the older box unchanged
    IBoundingBox<P> getMergedBox(P point);

    // merges in place
    IBoundingBox<P> addPoint(P point);

    // merges and keeops the older box unchaged
    IBoundingBox<P> getMergedBox(IBoundingBox<P> otherBox);

    // merges in place
    IBoundingBox<P> addBox(IBoundingBox<P> otherBox);

    BoundingBox copyBoxToDouble();

    float getMinValueFloat(int i);

    float getMaxValueFloat(int i);

}
