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

public class CompactLeaf {

    private final int pointIndex;
    private short parentIndex;
    private int mass;

    public CompactLeaf(int pointIndex, short parentIndex) {
        this.pointIndex = pointIndex;
        this.parentIndex = parentIndex;
        mass = 1;
    }

    public int getPointIndex() {
        return pointIndex;
    }

    public short getParentIndex() {
        return parentIndex;
    }

    public int getMass() {
        return mass;
    }

    public void incrementMass() {
        mass++;
    }

    public void decrementMass() {
        mass--;
    }

    public void setParentIndex(short parentIndex) {
        this.parentIndex = parentIndex;
    }
}
