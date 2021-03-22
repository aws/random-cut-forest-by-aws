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
import java.util.stream.Collectors;

import com.amazon.randomcutforest.ComponentList;

/**
 * Traverse the trees in a forest sequentially.
 * 
 * @param <PointReference> references to a point
 * @param <Point>          explicit data type of a point
 */
public class SequentialForestUpdateExecutor<PointReference, Point>
        extends AbstractForestUpdateExecutor<PointReference, Point> {

    public SequentialForestUpdateExecutor(IUpdateCoordinator<PointReference, Point> updateCoordinator,
            ComponentList<PointReference, Point> components) {
        super(updateCoordinator, components);
    }

    @Override
    protected List<UpdateResult<PointReference>> update(PointReference point, long seqNum) {
        return components.stream().map(t -> t.update(point, seqNum)).filter(UpdateResult::isStateChange)
                .collect(Collectors.toList());
    }
}
