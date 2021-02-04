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

package com.amazon.randomcutforest.sampler;

import java.util.List;
import java.util.Optional;

/**
 * <p>
 * A sampler that can be updated iteratively from a stream of data points. The
 * update operation is broken into two steps: an "accept" step and an "add"
 * step. During the "accept" step, the sampler decides whether to accept a new
 * point into sample. The decision rule will depend on the sampler
 * implementation If the sampler is full, accepting a new point requires the
 * sampler to evict a point currently in the sample. This operation is also part
 * of the accept step.
 * </p>
 *
 * <p>
 * If the outcome of the accept step is to accept a new point, then the sampler
 * continues to the second step to add a point to the sample (if the outcome is
 * not to accept a new point, then this step is not invoked). The reason for
 * this two-step process is because sampler update steps may be interleaved with
 * model update steps in
 * {@link com.amazon.randomcutforest.executor.IUpdatable#update} (see
 * {@link com.amazon.randomcutforest.executor.SamplerPlusTree#update}, for
 * example). In particular, if a new point is accepted into the sampler whose
 * value is equal to an existing point in the sample, then the model may choose
 * to increment the count on the existing point rather than allocate new storage
 * for the duplicate point.
 * </p>
 *
 * @param <P> The point type.
 */
public interface IStreamSampler<P> {
    /**
     * Submit a point to the sampler and return true if the point is accepted into
     * the sample. By default this method chains together the {@link #acceptPoint}
     * and {@link #addPoint} methods. If a point was evicted from the sample as a
     * side effect, then the evicted point will be available in
     * {@link #getEvictedPoint()} until the next call to {@link #addPoint}.
     *
     * @param point         The point submitted to the sampler.
     * @param sequenceIndex the sequence number
     * @return true if the point is accepted and added to the sample, false if the
     *         point is rejected.
     */
    default boolean update(P point, long sequenceIndex) {
        if (acceptPoint(sequenceIndex)) {
            addPoint(point);
            return true;
        }
        return false;
    }

    /**
     * This is the first step in a two-step sample operation. In this step, the
     * sampler makes a decision about whether to accept a new point into the sample.
     * If it decides to accept the point, then a new point can be added by calling
     * {@link #addPoint}.
     *
     * If a point needs to be evicted before a new point is added, eviction should
     * happen in this method. If a point is evicted during a call to
     * {@code acceptSample}, it will be available by calling
     * {@link #getEvictedPoint()} until the next time {@code acceptSample} is
     * called.
     *
     * @param sequenceIndex The sequence of the the point being submitted to the
     *                      sampler.
     * @return true if the point should be added to the sample.
     */
    boolean acceptPoint(long sequenceIndex);

    /**
     * This is the second step in a two-step sample operation. If the
     * {@link #acceptPoint} method was called and returned true, then this method
     * should be called to complete the sampling operation by adding the point to
     * the sample. If a call to {@code addPoint} is not preceded by a successful
     * call to {@code acceptPoint}, then it may fail with an
     * {@code IllegalStateException}.
     * 
     * @param point The point being added to the sample.
     */
    void addPoint(P point);

    /**
     * Return the list of sampled points.
     * 
     * @return the list of sampled points.
     */
    List<ISampled<P>> getSample();

    /**
     * @return the point that was evicted from the sample in the most recent call to
     *         {@link #acceptPoint}, or {@code Optional.empty()} if no point was
     *         evicted.
     */

    Optional<ISampled<P>> getEvictedPoint();

    /**
     * @return true if this sampler contains enough points to support the anomaly
     *         score computation, false otherwise. By default, this will
     */
    default boolean isReady() {
        return size() >= getCapacity() / 4;
    }

    /**
     * @return true if the sampler has reached it's full capacity, false otherwise.
     */
    default boolean isFull() {
        return size() >= getCapacity();
    }

    /**
     * @return the number of points contained by the sampler when full.
     */
    int getCapacity();

    /**
     * @return the number of points currently contained by the sampler.
     */
    int size();

    /**
     * changes the time dependent sampling on the fly
     * 
     * @param lambda the rate of sampling
     */
    void setLambda(double lambda);

    void setMaxSequenceIndex(long maxSequenceIndex);

    void setSequenceIndexOfMostRecentLambdaUpdate(long index);

}
