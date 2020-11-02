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
import java.util.Optional;

import com.amazon.randomcutforest.state.store.PointStoreDoubleState;

public class PointSequencer implements IUpdateCoordinator<double[]> {

    public PointSequencer() {
    }

    @Override
    public double[] initUpdate(double[] point, long seqNum) {
        return point;
    }

    @Override
    public void completeUpdate(List<Optional<UpdateReturn<double[]>>> updateResults, double[] initial) {

    }

    @Override
    public PointStoreDoubleState getPointStoreState() {
        return null;
    }

}
