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

import java.util.Arrays;

import lombok.Data;

import com.amazon.randomcutforest.store.PointStoreDouble;

@Data
public class PointStoreDoubleState {

    public double[] store;
    public short[] refCount;
    public int[] freeIndexes;

    public PointStoreDoubleState() {

    }

    /**
     * Takes a PointStoreDouble and stores the information. Note that
     * freeIndexPointer is an index into an array [0:capacity-1] and can be -1 when
     * everything is occupied.
     *
     * @param pointStoreDouble
     */
    public PointStoreDoubleState(PointStoreDouble pointStoreDouble) {
        store = Arrays.copyOf(pointStoreDouble.getStore(), pointStoreDouble.getStore().length);
        refCount = Arrays.copyOf(pointStoreDouble.getRefCount(), pointStoreDouble.getRefCount().length);
    }

}
