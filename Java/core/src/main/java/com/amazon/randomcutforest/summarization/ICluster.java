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

import com.amazon.randomcutforest.util.WeightedIndex;

public interface ICluster {

    // adding a point to a cluster
    void addPoint(int index, float weight);

    // restting statistics for a potential reassignment
    void reset();

    // average distance of a point from a cluster representative
    double average_radius();

    // weight computation
    double getWeight();

    // is a point well expressed by the cluster
    boolean captureBeforeReset(float[] point, BiFunction<float[], float[], Double> distance);

    // optimize the cluster representation based on assigned points
    void recompute(List<WeightedIndex<float[]>> points, BiFunction<float[], float[], Double> distance);

    // merge another cluster of same type
    void mergeInto(ICluster other, BiFunction<float[], float[], Double> distance);

    // distance of apoint from a cluster
    double distance(float[] point, BiFunction<float[], float[], Double> distance);

    // distance of another cluster from this cluster
    double distance(ICluster other, BiFunction<float[], float[], Double> distance);

    // a primary representative of the cluster
    float[] primaryRepresentative(BiFunction<float[], float[], Double> distance);

    // all potential representativess of a cluster
    List<WeightedIndex<float[]>> getRepresentatives();
}
