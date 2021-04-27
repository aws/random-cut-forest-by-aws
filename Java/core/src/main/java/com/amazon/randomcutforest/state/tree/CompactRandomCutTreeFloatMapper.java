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

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IContextualStateMapper;
import com.amazon.randomcutforest.state.store.NodeStoreMapper;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;

@Getter
@Setter
public class CompactRandomCutTreeFloatMapper implements
        IContextualStateMapper<CompactRandomCutTreeFloat, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    private boolean partialTreeInUse = false;
    private boolean compress = true;

    @Override
    public CompactRandomCutTreeFloat toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context,
            long seed) {

        INodeStore nodeStore;

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setUsePartialTrees(state.isPartialTreeInUse());
        nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());

        CompactRandomCutTreeFloat tree = new CompactRandomCutTreeFloat.Builder()
                .enableBoundingBoxCaching(state.isEnableCache())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndices()).maxSize(state.getMaxSize())
                .root(state.getRoot()).randomSeed(seed).pointStore((PointStoreFloat) context.getPointStore())
                .nodeStore(nodeStore).centerOfMassEnabled(state.isEnableCenterOfMass()).build();
        tree.setBoundingBoxCacheFraction(state.getBoundingBoxCacheFraction());
        return tree;
    }

    @Override
    public CompactRandomCutTreeState toState(CompactRandomCutTreeFloat model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        model.reorderNodesInBreadthFirstOrder();
        state.setRoot(model.getRoot());
        state.setMaxSize(model.getMaxSize());
        state.setPartialTreeInUse(model.enableSequenceIndices || partialTreeInUse);
        state.setEnableCache(model.enableCache);
        state.setStoreSequenceIndices(model.enableSequenceIndices);
        state.setEnableCenterOfMass(model.enableCenterOfMass);
        state.setBoundingBoxCacheFraction(model.getBoundingBoxCacheFraction());

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setCompress(compress);
        nodeStoreMapper.setUsePartialTrees(state.isPartialTreeInUse());
        nodeStoreMapper.setSinglePrecisionSet(true);
        state.setNodeStoreState(nodeStoreMapper.toState((NodeStore) model.getNodeStore()));

        return state;
    }
}
