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
import com.amazon.randomcutforest.state.store.LeafStoreMapper;
import com.amazon.randomcutforest.state.store.NodeStoreMapper;
import com.amazon.randomcutforest.state.store.SmallLeafStoreMapper;
import com.amazon.randomcutforest.state.store.SmallNodeStoreMapper;
import com.amazon.randomcutforest.store.ILeafStore;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.store.SmallLeafStore;
import com.amazon.randomcutforest.store.SmallNodeStore;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;

@Getter
@Setter
public class CompactRandomCutTreeFloatMapper implements
        IContextualStateMapper<CompactRandomCutTreeFloat, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    private boolean boundingBoxCacheEnabled;

    @Override
    public CompactRandomCutTreeFloat toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context,
            long seed) {

        INodeStore nodeStore;
        ILeafStore leafStore;
        if (context.getMaxSize() < SmallNodeStore.MAX_TREE_SIZE) {
            SmallLeafStoreMapper leafStoreMapper = new SmallLeafStoreMapper();
            leafStore = leafStoreMapper.toModel(state.getLeafStoreState());

            SmallNodeStoreMapper nodeStoreMapper = new SmallNodeStoreMapper();
            nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());
        } else {
            LeafStoreMapper leafStoreMapper = new LeafStoreMapper();
            leafStore = leafStoreMapper.toModel(state.getLeafStoreState());

            NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
            nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());
        }
        return new CompactRandomCutTreeFloat(context.getMaxSize(), seed, (PointStoreFloat) context.getPointStore(),
                leafStore, nodeStore, state.getRoot(), boundingBoxCacheEnabled);
    }

    @Override
    public CompactRandomCutTreeState toState(CompactRandomCutTreeFloat model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        state.setRoot(model.getRoot());

        if (model.getMaxSize() < SmallNodeStore.MAX_TREE_SIZE) {
            SmallLeafStoreMapper leafStoreMapper = new SmallLeafStoreMapper();
            state.setLeafStoreState(leafStoreMapper.toState((SmallLeafStore) model.getLeafStore()));

            SmallNodeStoreMapper nodeStoreMapper = new SmallNodeStoreMapper();
            state.setNodeStoreState(nodeStoreMapper.toState((SmallNodeStore) model.getNodeStore()));
        } else {
            LeafStoreMapper leafStoreMapper = new LeafStoreMapper();
            state.setLeafStoreState(leafStoreMapper.toState((LeafStore) model.getLeafStore()));

            NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
            state.setNodeStoreState(nodeStoreMapper.toState((NodeStore) model.getNodeStore()));
        }

        return state;
    }
}
