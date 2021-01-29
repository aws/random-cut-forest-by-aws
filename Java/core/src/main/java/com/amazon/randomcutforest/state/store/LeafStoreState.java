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

@Data
public class LeafStoreState {
    public int[] pointIndex;
    public short[] smallParentIndex;
    public short[] smallMass;
    public short[] smallFreeIndexes;

    public int[] parentIndex;
    public int[] mass;
    public int[] freeIndexes;
}
