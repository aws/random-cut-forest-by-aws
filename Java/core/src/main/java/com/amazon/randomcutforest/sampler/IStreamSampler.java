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

import com.amazon.randomcutforest.Sequential;

/**
 * A sampler that samples from an ordered sequence of points.
 * 
 * @param <P> The type representing the point value.
 */
public interface IStreamSampler<P> {
    /**
     * Submit a point to the sampler. The sampler implementation determines whether
     * the point is added to the sample or not.
     * 
     * @param point The point submitted to the sampler.
     * @return true if the point is accepted and added to the sample, false if the
     *         point is rejected.
     */
    boolean sample(Sequential<P> point);

    /**
     * @return the list of sequential points currently making up the sample.
     */
    List<Sequential<P>> getSamples();

    /**
     * @return the point that was evicted from the sample in the most recent call to
     *         {@link #sample}, or {@code Optional.empty()} if no point was evicted.
     */
    Optional<Sequential<P>> getEvictedPoint();
}
