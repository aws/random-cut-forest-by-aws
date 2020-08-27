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

package com.amazon.randomcutforest;

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

/**
 * A container type representing a value with an associated sequence index.
 * 
 * @param <T> The type contained value.
 */
public class Sequential<T> {

    protected final T value;
    protected final long sequenceIndex;

    /**
     * Create a new sequential object
     * 
     * @param value         The contained value.
     * @param sequenceIndex The sequence idnex.
     */
    public Sequential(T value, long sequenceIndex) {
        checkNotNull(value, "value must not be null");
        this.value = value;
        this.sequenceIndex = sequenceIndex;
    }

    /**
     * Copy constructor.
     * 
     * @param other An existing sequential value.
     */
    public Sequential(Sequential<T> other) {
        value = other.value;
        sequenceIndex = other.sequenceIndex;
    }

    /**
     * @return the contained vaule.
     */
    public T getValue() {
        return value;
    }

    /**
     * @return the sequence index
     */
    public long getSequenceIndex() {
        return sequenceIndex;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Sequential)) {
            return false;
        }

        Sequential<?> other = (Sequential<?>) o;
        return other.value.equals(value) && other.sequenceIndex == sequenceIndex;
    }
}
