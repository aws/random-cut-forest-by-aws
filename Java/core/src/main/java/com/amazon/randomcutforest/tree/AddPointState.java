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
    private short siblingOffset;
    private int cutDimension;
    private double cutValue;
    private IBox<P> savedBox;
    private boolean resolved;

    public void initialize(short sibling, short dim, double val, IBox<P> box) {
        siblingOffset = sibling;
        cutDimension = dim;
        cutValue = val;
        savedBox = box;
        resolved = false;
    }

    public void setResolved() {
        resolved = true;
    }

    public boolean getResolved() {
        return resolved;
    }

    public IBox<P> getSavedBox() {
        return savedBox;
    }

    public int getCutDimension() {
        return cutDimension;
    }

    public double getCutValue() {
        return cutValue;
    }

    public short getSiblingOffset() {
        return siblingOffset;
    }
}
