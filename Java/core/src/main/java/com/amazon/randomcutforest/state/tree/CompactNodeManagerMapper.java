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

import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.store.LeafStoreMapper;
import com.amazon.randomcutforest.state.store.NodeStoreMapper;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.tree.CompactNodeManager;

public class CompactNodeManagerMapper implements IStateMapper<CompactNodeManager, CompactNodeManagerState> {
    @Override
    public CompactNodeManagerState toState(CompactNodeManager manager) {
        CompactNodeManagerState state = new CompactNodeManagerState();

        LeafStoreMapper leafStoreMapper = new LeafStoreMapper();
        state.setLeafStoreState(leafStoreMapper.toState((LeafStore) manager.getLeafStore()));

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        state.setNodeStoreState(nodeStoreMapper.toState((NodeStore) manager.getNodeStore()));

        return state;
    }

    @Override
    public CompactNodeManager toModel(CompactNodeManagerState state, long seed) {
        LeafStoreMapper leafStoreMapper = new LeafStoreMapper();
        LeafStore leafStore = leafStoreMapper.toModel(state.getLeafStoreState());

        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        NodeStore nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState());

        return new CompactNodeManager(leafStore.getCapacity(), nodeStore, leafStore);
    }
}
