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

import com.amazon.randomcutforest.util.Weighted;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * a set of cunstions that a conceptual "cluster" should satisfy for any generic distance based clustering
 * where a distance function of type (R,R) -> double is provided externally. It is not feasible (short of
 * various assumptions) to check for the validity of a distance function and the clustering would not perform
 * any validity checks. The user is referred to https://en.wikipedia.org/wiki/Metric_(mathematics)
 *
 * It does not escape our attention that the clustering can use multiple different distance functions
 * over its execution. But such should be performed with caution.
 */
public interface ICluster<R> {

    // restting statistics for a potential reassignment
    void reset();

    // average distance of a point from a cluster representative
    double averageRadius();

    // weight computation
    double getWeight();

    // is a point well expressed by the cluster? To be used in the future.
    boolean captureBeforeReset(R point, BiFunction<R, R, Double> distance);

    // merge another cluster of same type
    void absorb(ICluster<R> other, BiFunction<R, R, Double> distance);

    // distance of apoint from a cluster
    double distance(R point, BiFunction<R,R, Double> distance);

    // distance of another cluster from this cluster
    double distance(ICluster<R> other, BiFunction<R, R, Double> distance);

    // a primary representative of the cluster
    R primaryRepresentative(BiFunction<R, R, Double> distance);

    // all potential representativess of a cluster
    List<Weighted<R>> getRepresentatives();

    List<Weighted<Integer>> getAssignedPoints();

    // optimize the cluster representation based on assigned points
    double recompute(Function<Integer,R> getPoint, BiFunction<R,R, Double> distance);

    // adding a point to a cluster
    void addPoint(int index, float weight, double dist, R point, BiFunction<R,R, Double> distance);


}
