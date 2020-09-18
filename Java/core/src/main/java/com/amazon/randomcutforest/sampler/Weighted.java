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

import com.amazon.randomcutforest.Sequential;

/**
 * A container class representing a weighted sequential value. This generic type
 * is used by {@link SimpleStreamSamplerV2} to store weighted points of
 * arbitrary type.
 * 
 * @param <P> The representation of the point value.
 */
public class Weighted<P> extends Sequential<P> {

    private final double weight;

    /**
     * Create a new weighted value from a sequential value.
     * 
     * @param seqPoint A sequential value.
     * @param weight   The weight value.
     */
    public Weighted(Sequential<P> seqPoint, double weight) {
        super(seqPoint);
        this.weight = weight;
    }

    /**
     * Create a new weighted value.
     * 
     * @param point         The point value.
     * @param sequenceIndex The sequence index for this weighted value.
     * @param weight        The weight value.
     */
    public Weighted(P point, long sequenceIndex, double weight) {
        super(point, sequenceIndex);
        this.weight = weight;
    }

    /**
     * @return the weight value.
     */
    public double getWeight() {
        return weight;
    }
}
