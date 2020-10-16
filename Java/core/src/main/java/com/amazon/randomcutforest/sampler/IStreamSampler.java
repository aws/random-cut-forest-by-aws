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

import com.amazon.randomcutforest.executor.Sequential;

/**
 * A sampler that samples from an ordered sequence of points.
 * 
 * @param <P> The type representing the points
 */
public interface IStreamSampler<P> {
    /**
     * Submit a point to the sampler. The sampler implementation determines whether
     * the point is added to the sample or not.
     *
     * This function is not used explicitly in RandomCutForest but is helpful in
     * testing the samplers.
     * 
     * @param point  The point submitted to the sampler.
     * @param seqNum the sequence number
     * @return true if the point is accepted and added to the sample, false if the
     *         point is rejected.
     */
    default boolean sample(P point, long seqNum) {
        Optional<Float> result = acceptSample(seqNum);
        if (result.isPresent()) {
            addSample(point, result.get(), seqNum);
            return true;
        }
        return false;
    }

    /**
     * the function that adds to the sampler
     * 
     * @param point  reference of point
     * @param weight weight value in sampler
     * @param seqNum the sequence number
     */
    void addSample(P point, float weight, long seqNum);

    /**
     * The function decides if the new object elem would be added to the queue.
     *
     * @param seqNum sequence number
     *
     * @return returns Optional.empty() if the entry is noe accepted; otherwise
     *         returns the weight
     */

    Optional<Float> acceptSample(long seqNum);

    /**
     * @return the list of weighted points currently making up the sample.
     */
    List<Weighted<P>> getWeightedSamples();

    /**
     * @return the list of Sequential points currently making up the sample. If the
     *         sequence number is not present then a dummy variable is added.
     */
    List<Sequential<P>> getSequentialSamples();

    /**
     * @return the point that was evicted from the sample in the most recent call to
     *         {@link #sample}, or {@code Optional.empty()} if no point was evicted.
     */

    Optional<Sequential<P>> getEvictedPoint();
}
