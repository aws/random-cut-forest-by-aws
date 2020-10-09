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
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;

import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.store.PointStoreDoubleData;

public class PointStoreCoordinator implements IUpdateCoordinator<Integer> {

    private final PointStoreDouble doubleStore;
    private final PointStore store;

    public PointStoreCoordinator(PointStoreDouble store) {
        checkNotNull(store, "store must not be null");
        this.doubleStore = store;
        this.store = null;
    }

    public PointStoreCoordinator(PointStore store) {
        checkNotNull(store, "store must not be null");
        this.doubleStore = null;
        this.store = store;
    }

    @Override
    public Integer initUpdate(double[] point, long seqNum) {
        if (doubleStore != null) {
            int pointIndex = doubleStore.add(point);
            return pointIndex;
        } else {
            int pointIndex = store.add((toFloatArray(point)));
            return pointIndex;
        }
    }

    @Override
    public void completeUpdate(List<Optional<UpdateReturn<Integer>>> updateResults, Integer updateInput) {
        updateResults.stream().filter(Optional::isPresent).map(Optional::get).forEach(result -> {
            doubleStore.incrementRefCount(result.getFirst());
            result.getSecond().ifPresent(index -> doubleStore.decrementRefCount(index));
        });
        doubleStore.decrementRefCount(updateInput);
    }

    @Override
    public PointStoreDoubleData getPointStoreData() {
        if (doubleStore != null) {
            return new PointStoreDoubleData(doubleStore);
        } else { // To be filled in
            return null;
        }
    }
}
