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

import static com.amazon.randomcutforest.state.Version.V2_0;

import lombok.Data;

import java.io.Serializable;

@Data
public class NodeStoreState implements Serializable {

    private String version = V2_0;

    private int capacity;
    private boolean compressed;
    private int[] cutDimension;
    private byte[] cutValueData;
    private String precision;
    private int root;

    private boolean canonicalAndNotALeaf;
    private int size;
    private int[] leftIndex;
    private int[] rightIndex;

    private int[] nodeFreeIndexes;
    private int nodeFreeIndexPointer;
    private int[] leafFreeIndexes;
    private int leafFreeIndexPointer;

    private boolean partialTreeStateEnabled;
    private int[] leafMass;
    private int[] leafPointIndex;

}
