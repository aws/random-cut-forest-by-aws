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

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.util.Weighted;

/**
 * a set of cunstions that a conceptual "cluster" should satisfy for any generic
 * distance based clustering where a distance function of type from (R,R) into
 * double is provided externally. It is not feasible (short of various
 * assumptions) to check for the validity of a distance function and the
 * clustering would not perform any validity checks. The user is referred to
 * https://en.wikipedia.org/wiki/Metric_(mathematics)
 *
 * It does not escape our attention that the clustering can use multiple
 * different distance functions over its execution. But such should be performed
 * with caution.
 */
public interface ICluster<R> {

    // restting statistics for a potential reassignment
    void reset();

    double averageRadius();

    // a measure of the noise/blur around a cluster; for single centroid clustering
    // this is the average distance of a point from a cluster representative
    double extentMeasure();

    // weight computation
    double getWeight();

    // merge another cluster of same type
    void absorb(ICluster<R> other, BiFunction<R, R, Double> distance);

    // distance of apoint from a cluster, has to be non-negative
    double distance(R point, BiFunction<R, R, Double> distance);

    // distance of another cluster from this cluster, has to be non negative
    double distance(ICluster<R> other, BiFunction<R, R, Double> distance);

    // all potential representativess of a cluster these are typically chosen to be
    // well scattered
    // by default the first entry is the primary representative
    List<Weighted<R>> getRepresentatives();

    // a primary representative of the cluster; by default it is the first in the
    // list of representatives
    // this additional function allows an option for optimization of runtime as well
    // as alternate
    // representations. For example the distance metric can be altered to be a fixed
    // linear combination
    // of the primary and secondary representatives, as in CURE
    // https://en.wikipedia.org/wiki/CURE_algorithm
    default R primaryRepresentative(BiFunction<R, R, Double> distance) {
        return getRepresentatives().get(0).index;
    }

    // Some of the algorithms, in particular the geometric ones may store the
    // assigned points for
    // iterative refinement. However that can be extremely inefficient if the
    // distance measure does not
    // have sufficient range, for example, string edit distances (for bounded
    // strings) are bounded in a
    // short interval. A soft assignment would create multiple copies of points (as
    // is appropriate) and
    // that can be significantly slower.
    default List<Weighted<Integer>> getAssignedPoints() {
        return Collections.emptyList();
    }

    // optimize the cluster representation based on assigned points; this is classic
    // iterative optimization
    // useful in EM type algorithms

    /**
     * optimize the cluster representation based on assigned points; this is classic
     * iterative optimization useful in EM type algorithms
     * 
     * @param getPoint a function that provides a point given an integer index
     * @param force    it set as true perform a slow and accurate recomputation;
     *                 otherwise approximation would suffice
     * @param distance the distance function
     * @return a measure of improvement (if any); this can be useful in the future
     *         as a part of the stopping condition
     */
    double recompute(Function<Integer, R> getPoint, boolean force, BiFunction<R, R, Double> distance);

    // adding a point to a cluster, and possibly updates the extent measure and the
    // assigned points
    void addPoint(int index, float weight, double dist, R point, BiFunction<R, R, Double> distance);

}
