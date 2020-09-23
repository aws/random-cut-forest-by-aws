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

import java.util.List;
import java.util.Optional;

/**
 * An IUpdateCoordinator is used in conjunction with a family of IUpdatable
 * instances. The coordinator transforms the input point into the form expected
 * by the updatable models, and processes the list of deleted points if needed.
 * An IUpdateCoordinator can be used to manage shared state.
 *
 * @param <P> An internal point representation.
 */
public interface IUpdateCoordinator<P> {
    /**
     * Transform the input point into a value that can be submitted to IUpdatable
     * instances.
     * 
     * @param point  The input point.
     * @param seqNum the sequence number; for future use
     * @return The point transformed into the representation expected by an
     *         IUpdatable instance.
     */
    P initUpdate(double[] point, long seqNum);

    /**
     * Complete the update. This method is called by IUpdateCoordinator after all
     * IUpdabale instances have completed their individual updates. This method
     * receives the list of points that were deleted IUpdatable instances for
     * further processing if needed.
     * 
     * @param updateResults A list of points that were deleted.
     */
    void completeUpdate(List<Optional<UpdateReturn<P>>> updateResults);

}
