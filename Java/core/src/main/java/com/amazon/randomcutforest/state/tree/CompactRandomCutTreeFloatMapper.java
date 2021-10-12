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

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IContextualStateMapper;
import com.amazon.randomcutforest.state.store.NodeStoreMapper;
import com.amazon.randomcutforest.state.store.SmallNodeStoreMapper;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.store.SmallNodeStore;
import com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;

@Getter
@Setter
public class CompactRandomCutTreeFloatMapper implements
        IContextualStateMapper<CompactRandomCutTreeFloat, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    private boolean partialTreeStateEnabled = false;
    private boolean compressed = true;

    @Override
    public CompactRandomCutTreeFloat toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context,
            long seed) {

        INodeStore nodeStore;

        if (AbstractCompactRandomCutTree.canUseSmallNodeStore(
                Precision.valueOf(state.getNodeStoreState().getPrecision()), state.getMaxSize(),
                state.getDimensions())) {
            SmallNodeStoreMapper nodeStoreMapper = new SmallNodeStoreMapper();
            nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
            nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());
        } else {
            NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
            nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
            nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());
        }

        CompactRandomCutTreeFloat tree = new CompactRandomCutTreeFloat.Builder()
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).maxSize(state.getMaxSize())
                .root(state.getRoot()).randomSeed(state.getSeed()).pointStore((PointStoreFloat) context.getPointStore())
                .nodeStore(nodeStore).centerOfMassEnabled(state.isCenterOfMassEnabled())
                .outputAfter(state.getOutputAfter()).build();
        return tree;
    }

    @Override
    public CompactRandomCutTreeState toState(CompactRandomCutTreeFloat model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        model.reorderNodesInBreadthFirstOrder();
        state.setRoot(model.getRootIndex());
        state.setMaxSize(model.getMaxSize());
        state.setPartialTreeState(model.storeSequenceIndexesEnabled || partialTreeStateEnabled);
        state.setStoreSequenceIndexesEnabled(model.storeSequenceIndexesEnabled);
        state.setCenterOfMassEnabled(model.centerOfMassEnabled);
        state.setBoundingBoxCacheFraction(model.getBoundingBoxCacheFraction());
        state.setOutputAfter(model.getOutputAfter());
        state.setSeed(model.getRandomSeed());
        state.setDimensions(model.getDimension());

        if (model.isSmallNodeStoreInUse()) {
            SmallNodeStoreMapper nodeStoreMapper = new SmallNodeStoreMapper();
            nodeStoreMapper.setCompressionEnabled(compressed);
            nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
            nodeStoreMapper.setPrecision(Precision.FLOAT_32);
            state.setNodeStoreState(nodeStoreMapper.toState((SmallNodeStore) model.getNodeStore()));
        } else {
            NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
            nodeStoreMapper.setCompressionEnabled(compressed);
            nodeStoreMapper.setPartialTreeStateEnabled(state.isPartialTreeState());
            nodeStoreMapper.setPrecision(Precision.FLOAT_32);
            state.setNodeStoreState(nodeStoreMapper.toState((NodeStore) model.getNodeStore()));
        }

        return state;
    }
}
