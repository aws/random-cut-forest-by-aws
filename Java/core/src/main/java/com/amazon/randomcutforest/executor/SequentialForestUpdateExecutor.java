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

package com.amazon.randomcutforest.executor;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.ComponentList;

/**
 * Traverse the trees in a forest sequentially.
 */
public class SequentialForestUpdateExecutor<P> extends AbstractForestUpdateExecutor<P> {

    public SequentialForestUpdateExecutor(IUpdateCoordinator<P> updateCoordinator, ComponentList<P> components) {
        super(updateCoordinator, components);
    }

    @Override
    protected List<Optional<UpdateReturn<P>>> update(P point, long seqNum) {
        return components.stream().map(t -> t.update(point, seqNum)).collect(Collectors.toList());
    }
}
