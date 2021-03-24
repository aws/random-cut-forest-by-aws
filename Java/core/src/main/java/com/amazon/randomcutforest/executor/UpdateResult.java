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

import java.util.Optional;

import lombok.Builder;

/**
 * When {@link IUpdatable#update} is called, an updatable model may choose to
 * update its state with the submitted point. This class contains the result of
 * such an operation. A list of {@code AddPointResults}s is consumed by
 * {@link com.amazon.randomcutforest.IStateCoordinator#completeUpdate} to update
 * global state as needed to reflect the updates to individual component models.
 * 
 * @param <P> The point reference type.
 */
@Builder
public class UpdateResult<P> {

    private static final UpdateResult<Object> NOOP = builder().build();

    private final P addedPoint;

    private final P deletedPoint;

    /**
     * Return an {@code UpdateResult} value a no-op (an operation that did not
     * change the state of the model). For the returned value,
     * {@code isStateChange()} will be false.
     * 
     * @param <Q> The point reference type.
     * @return an {@code UpdateResult} value representing a no-op.
     */
    public static <Q> UpdateResult<Q> noop() {
        return (UpdateResult<Q>) NOOP;
    }

    /**
     * An optional containing a reference to the point that was added to the model
     * as part of the udpate call, or {@code Optional.empty()} if no point was
     * added.
     * 
     * @return an optional containing a reference to the point that was added to the
     *         model as part of the udpate call, or {@code Optional.empty()} if no
     *         point was added.
     */
    public Optional<P> getAddedPoint() {
        return Optional.ofNullable(addedPoint);
    }

    /**
     * Once a model is at capacity, a point may be deleted from the model as part of
     * an update. If a point is deleted during the update operation, then the
     * deleted point reference will be present in the result of this method.
     * 
     * @return a reference to the deleted point reference or
     *         {@code Optional.empty()} if no point was deleted.
     */
    public Optional<P> getDeletedPoint() {
        return Optional.ofNullable(deletedPoint);
    }

    /**
     * Return true if this update result represents a change to the updatable model.
     * A change means that a point was added to the model, and possibly a point was
     * deleted from the model.
     * 
     * @return true if this update result represents a change to the updatabla
     *         model.
     */
    public boolean isStateChange() {
        return addedPoint != null || deletedPoint != null;
    }
}
