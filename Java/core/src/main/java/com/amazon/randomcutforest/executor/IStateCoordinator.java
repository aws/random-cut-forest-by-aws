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

import com.amazon.randomcutforest.store.IPointStore;

/**
 * An IStateCoordinator is used in conjunction with a family of IUpdatable
 * instances. The coordinator transforms the input point into the form expected
 * by the updatable models, and processes the list of deleted points if needed.
 * An IStateCoordinator can be used to manage shared state.
 *
 * @param <PointReference> An internal point representation.
 * @param <Point>          Explicit point type
 */
public interface IStateCoordinator<PointReference, Point> {
    /**
     * Transform the input point into a value that can be submitted to IUpdatable
     * instances.
     *
     * @param point          The input point.
     * @param sequenceNumber the sequence number associated with the point
     * @return The point transformed into the representation expected by an
     *         IUpdatable instance.
     */
    PointReference initUpdate(double[] point, long sequenceNumber);

    /**
     * Complete the update. This method is called by IStateCoordinator after all
     * IUpdabale instances have completed their individual updates. This method
     * receives the list of points that were deleted IUpdatable instances for
     * further processing if needed.
     *
     * @param updateResults A list of points that were deleted.
     * @param updateInput   The corresponding output from {@link #initUpdate}, which
     *                      was passed into the update method for each component
     */
    void completeUpdate(List<UpdateResult<PointReference>> updateResults, PointReference updateInput);

    long getTotalUpdates();

    void setTotalUpdates(long totalUpdates);

    default IPointStore<Point> getStore() {
        return null;
    }
}
