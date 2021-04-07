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

import java.util.Arrays;
import java.util.List;

import lombok.Getter;

import com.amazon.randomcutforest.ComponentList;

/**
 * The class transforms input points into the form expected by internal models,
 * and submits transformed points to individual models for updating.
 *
 * @param <PointReference> The point representation used by model data
 *                         structures.
 * @param <Point>          The explicit data type of exchanging points
 */
@Getter
public abstract class AbstractForestUpdateExecutor<PointReference, Point> {

    protected final IStateCoordinator<PointReference, Point> updateCoordinator;
    protected final ComponentList<PointReference, Point> components;

    /**
     * Create a new AbstractForestUpdateExecutor.
     * 
     * @param updateCoordinator The update coordinater that will be used to
     *                          transform points and process deleted points if
     *                          needed.
     * @param components        A list of models to update.
     */
    protected AbstractForestUpdateExecutor(IStateCoordinator<PointReference, Point> updateCoordinator,
            ComponentList<PointReference, Point> components) {
        this.updateCoordinator = updateCoordinator;
        this.components = components;
    }

    /**
     * Update the forest with the given point. The point is submitted to each
     * sampler in the forest. If the sampler accepts the point, the point is
     * submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        update(point, updateCoordinator.getTotalUpdates());
    }

    public void update(double[] point, long sequenceNumber) {
        double[] pointCopy = cleanCopy(point);
        PointReference updateInput = updateCoordinator.initUpdate(pointCopy, sequenceNumber);
        List<UpdateResult<PointReference>> results = update(updateInput, sequenceNumber);
        updateCoordinator.completeUpdate(results, updateInput);
    }

    /**
     * Internal update method which submits the given input value to
     * {@link IUpdatable#update} for each model managed by this executor.
     *
     * @param updateInput  Input value that will be submitted to the update method
     *                     for each tree.
     * @param currentIndex the timestamp
     * @return a list of points that were deleted from the model as part of the
     *         update.
     */
    protected abstract List<UpdateResult<PointReference>> update(PointReference updateInput, long currentIndex);

    /**
     * Returns a clean deep copy of the point.
     *
     * Current clean-ups include changing negative zero -0.0 to positive zero 0.0.
     *
     * @param point The original data point.
     * @return a clean deep copy of the original point.
     */
    protected double[] cleanCopy(double[] point) {
        double[] pointCopy = Arrays.copyOf(point, point.length);
        for (int i = 0; i < point.length; i++) {
            if (pointCopy[i] == 0.0) {
                pointCopy[i] = 0.0;
            }
        }
        return pointCopy;
    }
}
