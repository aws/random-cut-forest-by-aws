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
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeDouble;

@Getter
@Setter
public class CompactRandomCutTreeDoubleMapper implements
        IContextualStateMapper<CompactRandomCutTreeDouble, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    private boolean partialTreeStateEnabled = false;
    private boolean compress = true;

    @Override
    public CompactRandomCutTreeDouble toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context,
            long seed) {

        INodeStore nodeStore;

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
        nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());

        CompactRandomCutTreeDouble tree = new CompactRandomCutTreeDouble.Builder()
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).maxSize(state.getMaxSize())
                .root(state.getRoot()).randomSeed(state.getSeed())
                .pointStore((PointStoreDouble) context.getPointStore()).nodeStore(nodeStore)
                .centerOfMassEnabled(state.isCenterOfMassEnabled()).outputAfter(state.getOutputAfter()).build();
        return tree;

    }

    @Override
    public CompactRandomCutTreeState toState(CompactRandomCutTreeDouble model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        model.reorderNodesInBreadthFirstOrder();
        state.setMaxSize(model.getMaxSize());
        state.setRoot(model.getRootIndex());
        state.setPartialTreeState(model.storeSequenceIndexesEnabled || partialTreeStateEnabled);
        state.setStoreSequenceIndexesEnabled(model.storeSequenceIndexesEnabled);
        state.setCenterOfMassEnabled(model.centerOfMassEnabled);
        state.setOutputAfter(model.getOutputAfter());
        state.setSeed(model.getRandomSeed());

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setCompressionEnabled(compress);
        nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
        state.setNodeStoreState(nodeStoreMapper.toState((NodeStore) model.getNodeStore()));
        return state;
    }
}
