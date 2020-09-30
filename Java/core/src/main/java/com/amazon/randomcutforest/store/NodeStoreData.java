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

package com.amazon.randomcutforest.store;

import java.util.Arrays;

public class NodeStoreData {
    public final short[] parentIndex;
    public final short[] leftIndex;
    public final short[] rightIndex;
    public final int[] cutDimension;
    public final double[] cutValue;
    public final int[] mass;
    public final short[] freeIndexes;

    public NodeStoreData(NodeStore nodeStore) {
        leftIndex = Arrays.copyOf(nodeStore.leftIndex, nodeStore.leftIndex.length);
        rightIndex = Arrays.copyOf(nodeStore.rightIndex, nodeStore.rightIndex.length);
        parentIndex = Arrays.copyOf(nodeStore.parentIndex, nodeStore.parentIndex.length);
        mass = Arrays.copyOf(nodeStore.mass, nodeStore.mass.length);
        cutDimension = Arrays.copyOf(nodeStore.cutDimension, nodeStore.cutDimension.length);
        cutValue = Arrays.copyOf(nodeStore.cutValue, nodeStore.cutValue.length);
        freeIndexes = Arrays.copyOf(nodeStore.freeIndexes, nodeStore.freeIndexPointer + 1);
    }

}
