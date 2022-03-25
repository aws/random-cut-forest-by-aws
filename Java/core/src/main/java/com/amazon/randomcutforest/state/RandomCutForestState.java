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

package com.amazon.randomcutforest.state;

import static com.amazon.randomcutforest.state.Version.V3_0;

import java.util.List;

import lombok.Data;

import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeState;

/**
 * A class that encapsulates most of the data used in a RandomCutForest such
 * that the forest can be serialized and deserialized.
 */
@Data
public class RandomCutForestState {

    private String version = V3_0;

    private long totalUpdates;

    private double timeDecay;

    private int numberOfTrees;

    private int sampleSize;

    private int shingleSize;

    private int dimensions;

    private int outputAfter;

    private boolean compressed;

    private boolean partialTreeState;

    private double boundingBoxCacheFraction;

    private boolean storeSequenceIndexesEnabled;

    private boolean compact;

    private boolean internalShinglingEnabled;

    private boolean centerOfMassEnabled;

    private String precision;

    private PointStoreState pointStoreState;

    private List<CompactSamplerState> compactSamplerStates;

    private List<CompactRandomCutTreeState> compactRandomCutTreeStates;

    private ExecutionContext executionContext;

    // Mapper options

    private boolean saveTreeStateEnabled;

    private boolean saveSamplerStateEnabled;

    private boolean saveCoordinatorStateEnabled;

}
