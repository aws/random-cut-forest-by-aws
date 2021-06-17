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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.List;

import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStore;

/**
 * pointstore coordinator for compact RCF
 * 
 * @param <Point> the datatype of the actual point
 */

public class PointStoreCoordinator<Point> extends AbstractUpdateCoordinator<Integer, Point> {

    private final IPointStore<Point> store;

    public PointStoreCoordinator(IPointStore<Point> store) {
        checkNotNull(store, "store must not be null");
        this.store = store;
    }

    @Override
    public Integer initUpdate(double[] point, long sequenceNumber) {
        int index = store.add(point, sequenceNumber);
        return (index == PointStore.INFEASIBLE_POINTSTORE_INDEX) ? null : index;
    }

    @Override
    public void completeUpdate(List<UpdateResult<Integer>> updateResults, Integer updateInput) {
        if (updateInput != null) { // can be null for initial shingling
            updateResults.forEach(result -> {
                result.getAddedPoint().ifPresent(store::incrementRefCount);
                result.getDeletedPoint().ifPresent(store::decrementRefCount);
            });
            store.decrementRefCount(updateInput);
        }
        totalUpdates++;
    }

    public IPointStore<Point> getStore() {
        return store;
    }
}
