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

/**
 * A minimal implementation of {@link IUpdateCoordinator} that does not
 * transform the input point.
 */
public class PassThroughCoordinator extends AbstractUpdateCoordinator<double[]> {
    /**
     * Return the input point without making a copy.
     * 
     * @param point The input point.
     * @return the input point.
     */
    @Override
    public double[] initUpdate(double[] point) {
        return point;
    }

    /**
     * Increment the totalUpdates counter. The method arguments are not used.
     * 
     * @param updateResults A list of points that were deleted.
     * @param updateInput   The corresponding output from {@link #initUpdate}, which
     *                      was passed into the update method for each component
     */
    @Override
    public void completeUpdate(List<UpdateResult<double[]>> updateResults, double[] updateInput) {
        totalUpdates++;
    }
}
