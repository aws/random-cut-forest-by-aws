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

package com.amazon.randomcutforest.state.store;

import lombok.Data;

/**
 * A class for storing the state of a
 * {@link com.amazon.randomcutforest.store.PointStoreDouble} or a
 * {@link com.amazon.randomcutforest.store.PointStoreFloat}. Depending on which
 * kind of point store was serialized, one of the fields {@code doubleData} or
 * {@code floatData} will be null.
 */
@Data
public class PointStoreState {
    /**
     * Point data for a {@link com.amazon.randomcutforest.store.PointStoreDouble}.
     */
    private double[] doubleData;
    /**
     * Point data for a {@link com.amazon.randomcutforest.store.PointStoreFloat}.
     */
    private float[] floatData;
    /**
     * An array of reference counts for each stored point.
     */
    private short[] refCount;
    /**
     * An array of indexes indicating which data slots are occupied.
     */
    private int[] freeIndexes;
}
