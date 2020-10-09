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

import com.amazon.randomcutforest.sampler.Weighted;

/**
 * A container type representing a value with an associated sequence index.
 * 
 * @param <P> The type contained value.
 */
public class Sequential<P> extends Weighted<P> {

    protected final long sequenceIndex;

    /**
     * Create a new sequential object
     * 
     * @param value         The contained value.
     * @param weight        The weight for Weighted
     * @param sequenceIndex The sequence idnex.
     */
    public Sequential(P value, double weight, long sequenceIndex) {
        super(value, weight);
        this.sequenceIndex = sequenceIndex;
    }

    /*
     * public Sequential(P value, long sequenceIndex, double weight) { super(value,
     * weight); this.sequenceIndex = sequenceIndex; }
     */

    /**
     * Copy constructor.
     * 
     * @param other An existing sequential value.
     */
    public Sequential(Sequential<P> other) {
        super(other.getValue(), other.getWeight());
        sequenceIndex = other.sequenceIndex;
    }

    /**
     * @return the sequence index
     */
    public long getSequenceIndex() {
        return sequenceIndex;
    }
}
