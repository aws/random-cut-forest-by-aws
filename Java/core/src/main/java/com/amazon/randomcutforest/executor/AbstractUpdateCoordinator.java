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

/**
 * A partial implementation of the {@link IUpdateCoordinator} interface that
 * defines a protected instance variable to track total updates and implements
 * the {@link IUpdateCoordinator#getTotalUpdates()} method. Classes that extend
 * AbstractUpdateCoordinator are responsible for incrementing the totalUpdates
 * counter after completing an update successfully.
 *
 * @param <P> An internal point representation.
 */
public abstract class AbstractUpdateCoordinator<P> implements IUpdateCoordinator<P> {
    protected long totalUpdates;

    public AbstractUpdateCoordinator(long totalUpdates) {
        this.totalUpdates = totalUpdates;
    }

    public AbstractUpdateCoordinator() {
        this(0L);
    }

    @Override
    public long getTotalUpdates() {
        return totalUpdates;
    }

    @Override
    public void setTotalUpdates(long totalUpdates) {
        this.totalUpdates = totalUpdates;
    }
}
