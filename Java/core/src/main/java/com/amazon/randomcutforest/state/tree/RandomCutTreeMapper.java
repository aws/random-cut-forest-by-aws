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

import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.state.IContextualStateMapper;
import com.amazon.randomcutforest.state.Version;
import com.amazon.randomcutforest.tree.AbstractNodeStore;
import com.amazon.randomcutforest.tree.RandomCutTree;

@Getter
@Setter
public class RandomCutTreeMapper
        implements IContextualStateMapper<RandomCutTree, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    @Override
    public RandomCutTree toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context, long seed) {

        AbstractNodeStoreMapper nodeStoreMapper = new AbstractNodeStoreMapper();
        nodeStoreMapper.setRoot(state.getRoot());
        AbstractNodeStore nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState(), context);

        int dimension = (state.getDimensions() != 0) ? state.getDimensions() : context.getPointStore().getDimensions();
        // boundingBoxcache is not set deliberately;
        // it should be set after the partial tree is complete
        RandomCutTree tree = new RandomCutTree.Builder().dimension(dimension)
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).capacity(state.getMaxSize())
                .setRoot(state.getRoot()).randomSeed(state.getSeed()).pointStoreView(context.getPointStore())
                .nodeStore(nodeStore).centerOfMassEnabled(state.isCenterOfMassEnabled())
                .outputAfter(state.getOutputAfter()).build();
        return tree;
    }

    @Override
    public CompactRandomCutTreeState toState(RandomCutTree model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        state.setVersion(Version.V3_0);
        int root = model.getRoot();
        AbstractNodeStoreMapper nodeStoreMapper = new AbstractNodeStoreMapper();
        nodeStoreMapper.setRoot(root);
        state.setNodeStoreState(nodeStoreMapper.toState(model.getNodeStore()));
        // the compression of nodeStore would change the root
        if ((root != Null) && (root < model.getNumberOfLeaves() - 1)) {
            root = 0; // reordering is forced
        }
        state.setRoot(root);
        state.setMaxSize(model.getNumberOfLeaves());
        state.setPartialTreeState(true);
        state.setStoreSequenceIndexesEnabled(model.isStoreSequenceIndexesEnabled());
        state.setCenterOfMassEnabled(model.isCenterOfMassEnabled());
        state.setBoundingBoxCacheFraction(model.getBoundingBoxCacheFraction());
        state.setOutputAfter(model.getOutputAfter());
        state.setSeed(model.getRandomSeed());
        state.setDimensions(model.getDimension());

        return state;
    }
}
