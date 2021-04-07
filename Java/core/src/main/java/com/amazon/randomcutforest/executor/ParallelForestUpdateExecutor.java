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
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import com.amazon.randomcutforest.ComponentList;

/**
 * An implementation of forest traversal methods that uses a private thread pool
 * to visit trees in parallel.
 * 
 * @param <PointReference> references to a point
 * @param <Point>          explicit data type of a point
 */
public class ParallelForestUpdateExecutor<PointReference, Point>
        extends AbstractForestUpdateExecutor<PointReference, Point> {

    private ForkJoinPool forkJoinPool;
    private final int threadPoolSize;

    public ParallelForestUpdateExecutor(IStateCoordinator<PointReference, Point> updateCoordinator,
            ComponentList<PointReference, Point> components, int threadPoolSize) {
        super(updateCoordinator, components);
        this.threadPoolSize = threadPoolSize;
        forkJoinPool = new ForkJoinPool(threadPoolSize);
    }

    @Override
    protected List<UpdateResult<PointReference>> update(PointReference point, long seqNum) {
        return submitAndJoin(() -> components.parallelStream().map(t -> t.update(point, seqNum))
                .filter(UpdateResult::isStateChange).collect(Collectors.toList()));
    }

    private <T> T submitAndJoin(Callable<T> callable) {
        if (forkJoinPool == null) {
            forkJoinPool = new ForkJoinPool(threadPoolSize);
        }
        return forkJoinPool.submit(callable).join();
    }
}
