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
import com.amazon.randomcutforest.tree.AbstractNodeStore;
import com.amazon.randomcutforest.tree.NewRandomCutTree;

@Getter
@Setter
public class RandomCutTreeMapper
        implements IContextualStateMapper<NewRandomCutTree, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    @Override
    public NewRandomCutTree toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context, long seed) {

        AbstractNodeStoreMapper nodeStoreMapper = new AbstractNodeStoreMapper();
        nodeStoreMapper.setRoot(state.getRoot());
        AbstractNodeStore nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState(), context);

        NewRandomCutTree tree = new NewRandomCutTree.Builder()
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction())
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).capacity(state.getMaxSize())
                .setRoot(state.getRoot()).randomSeed(state.getSeed()).pointStoreView(context.getPointStore())
                .nodeStore(nodeStore).centerOfMassEnabled(state.isCenterOfMassEnabled())
                .outputAfter(state.getOutputAfter()).build();
        return tree;
    }

    @Override
    public CompactRandomCutTreeState toState(NewRandomCutTree model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        // model.reorderNodesInBreadthFirstOrder();
        state.setRoot(model.getRoot());
        state.setMaxSize(model.getNumberOfLeaves());
        state.setPartialTreeState(true);
        state.setStoreSequenceIndexesEnabled(model.isStoreSequenceIndexesEnabled());
        state.setCenterOfMassEnabled(model.isCenterOfMassEnabled());
        state.setBoundingBoxCacheFraction(model.getBoundingBoxCacheFraction());
        state.setOutputAfter(model.getOutputAfter());
        state.setSeed(model.getRandomSeed());
        state.setDimensions(model.getDimension());

        AbstractNodeStoreMapper nodeStoreMapper = new AbstractNodeStoreMapper();
        state.setNodeStoreState(nodeStoreMapper.toState(model.getNodeStore()));

        return state;
    }
}
