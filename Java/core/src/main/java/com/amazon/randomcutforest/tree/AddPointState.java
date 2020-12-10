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

public class AddPointState<P> {
    private int siblingOffset;
    private int cutDimension;
    private double cutValue;
    private IBoundingBox<P> savedBox;
    private IBoundingBox<P> currentBox;
    private boolean resolved;
    private final int pointIndex;

    public void initialize(int sibling, int dim, double val, IBoundingBox<P> box) {
        siblingOffset = sibling;
        cutDimension = dim;
        cutValue = val;
        savedBox = box;
        currentBox = box;
        resolved = false;
    }

    public AddPointState(int pointIndex) {
        this.pointIndex = pointIndex;
    }

    public int getPointIndex() {
        return pointIndex;
    }

    public void setResolved() {
        resolved = true;
    }

    public boolean getResolved() {
        return resolved;
    }

    public IBoundingBox<P> getSavedBox() {
        return savedBox;
    }

    public int getCutDimension() {
        return cutDimension;
    }

    public double getCutValue() {
        return cutValue;
    }

    public int getSiblingOffset() {
        return siblingOffset;
    }

    public IBoundingBox<P> getCurrentBox() {
        return currentBox;
    }

    public void setCurrentBox(IBoundingBox<P> currentBox) {
        this.currentBox = currentBox;
    }
}
