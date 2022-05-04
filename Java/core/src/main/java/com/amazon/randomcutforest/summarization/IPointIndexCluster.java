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

package com.amazon.randomcutforest.summarization;

import java.util.List;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.util.Weighted;

/**
 * an interface that defines a point index based implementation, where a
 * list/block of points are stored separately in a list (in the future,
 * pointstore) and the index provides a handle to specific point (under the
 * assumption that the list/poinstore) is not changed.
 */
public interface IPointIndexCluster extends ICluster {

    // adding a point to a cluster
    void addPoint(int index, float weight, double distance);

    // optimize the cluster representation based on assigned points
    double recompute(List<Weighted<float[]>> points, BiFunction<float[], float[], Double> distance);

}
