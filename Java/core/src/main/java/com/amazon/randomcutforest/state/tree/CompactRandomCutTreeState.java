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

package com.amazon.randomcutforest.state.tree;

import static com.amazon.randomcutforest.state.Version.V2_0;

import java.io.Serializable;

import lombok.Data;

import com.amazon.randomcutforest.state.store.NodeStoreState;

@Data
public class CompactRandomCutTreeState implements Serializable {
    private static final long serialVersionUID = 1L;

    private String version = V2_0;
    private int root;
    private int maxSize;
    private int outputAfter;
    private boolean storeSequenceIndexesEnabled;
    private boolean centerOfMassEnabled;
    private NodeStoreState nodeStoreState;
    private double boundingBoxCacheFraction;
    private boolean partialTreeState;
    private long seed;
    private int id;
    private int dimensions;
    private long staticSeed;
    private float weight;
    private byte[] auxiliaryData;
    private boolean hasAuxiliaryData;

}
