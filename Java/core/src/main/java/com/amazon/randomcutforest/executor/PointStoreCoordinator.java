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

package com.amazon.randomcutforest;

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.store.PointStore;

public class PointStoreCoordinator implements IUpdateCoordinator<Integer> {

    private final PointStore store;
    private long currentIndex;

    public PointStoreCoordinator(PointStore store) {
        checkNotNull(store, "store must not be null");
        this.store = store;
        currentIndex = 1L;
    }

    @Override
    public Integer initUpdate(double[] point, long seqNum) {
        int pointIndex = store.add(CommonUtils.toFloatArray(point));
        return pointIndex;
    }

    @Override
    public void completeUpdate(List<Optional<UpdateReturn<Integer>>> updateResults) {
        updateResults.stream().filter(Optional::isPresent).map(Optional::get).forEach(result -> {
            store.incrementRefCount(result.getFirst());
            result.getSecond().ifPresent(index -> store.decrementRefCount(index));
        });
        currentIndex++;
    }

}
