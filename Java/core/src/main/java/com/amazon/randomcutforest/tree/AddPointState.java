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

public class AddPointState<Point, NodeReference, PointReference> {
    private NodeReference siblingOffset;
    private int cutDimension;
    private double cutValue;
    private AbstractBoundingBox<Point> savedBox;
    private AbstractBoundingBox<Point> currentBox;
    private boolean resolved;
    private final PointReference pointIndex;
    private long sequenceNumber;

    public void initialize(NodeReference sibling, int dim, double val, long sequenceNumber,
            AbstractBoundingBox<Point> mergedBox, AbstractBoundingBox<Point> siblingBox) {
        siblingOffset = sibling;
        cutDimension = dim;
        cutValue = val;
        this.savedBox = mergedBox;
        this.currentBox = siblingBox;
        resolved = false;
        this.sequenceNumber = sequenceNumber;
    }

    public AddPointState(PointReference pointIndex) {
        this.pointIndex = pointIndex;
    }

    public PointReference getPointIndex() {
        return pointIndex;
    }

    public void setResolved() {
        resolved = true;
    }

    public boolean getResolved() {
        return resolved;
    }

    public AbstractBoundingBox<Point> getSavedBox() {
        return savedBox;
    }

    public int getCutDimension() {
        return cutDimension;
    }

    public double getCutValue() {
        return cutValue;
    }

    public NodeReference getSiblingOffset() {
        return siblingOffset;
    }

    public AbstractBoundingBox<Point> getCurrentBox() {
        return currentBox;
    }

    public void setCurrentBox(AbstractBoundingBox<Point> currentBox) {
        this.currentBox = currentBox;
    }

    public long getSequenceNumber() {
        return sequenceNumber;
    }
}
