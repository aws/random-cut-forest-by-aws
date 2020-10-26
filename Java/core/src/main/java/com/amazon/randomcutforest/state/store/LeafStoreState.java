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

import com.amazon.randomcutforest.store.LeafStore;

@Data
public class LeafStoreState {

    public int[] pointIndex;
    public short[] parentIndex;
    public int[] mass;
    public short[] freeIndexes;

    public LeafStoreState() {
    }

    public LeafStoreState(LeafStore leafStore) {
        pointIndex = Arrays.copyOf(leafStore.pointIndex, leafStore.pointIndex.length);
        parentIndex = Arrays.copyOf(leafStore.parentIndex, leafStore.parentIndex.length);
        mass = Arrays.copyOf(leafStore.mass, leafStore.mass.length);
        freeIndexes = Arrays.copyOf(leafStore.getFreeIndexes(), leafStore.getFreeIndexPointer() + 1);
    }
}
