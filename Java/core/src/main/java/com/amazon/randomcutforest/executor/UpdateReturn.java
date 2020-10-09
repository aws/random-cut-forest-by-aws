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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Optional;

/**
 * A container type representing a value with an associated sequence index.
 * 
 * @param <T> The type contained value.
 */
public class UpdateReturn<T> {

    protected final T first;
    protected final Optional<T> second;

    /**
     * Create a new return object
     *
     * @param first  The first value, cannot be null. This is the handle of an
     *               inserted point.
     * @param second The second value, the handle of any point evicted from a
     *               sampler. Can be null.
     */
    public UpdateReturn(T first, Optional<T> second) {
        checkNotNull(first, "first value must not be null");
        this.first = first;
        this.second = second;
    }

    /**
     * Copy constructor.
     * 
     * @param other An existing UpdateReturn value.
     */
    public UpdateReturn(UpdateReturn<T> other) {
        first = other.first;
        second = other.second;
    }

    /**
     * @return the first value
     */
    public T getFirst() {
        return first;
    }

    /**
     * @return the second value
     */
    public Optional<T> getSecond() {
        return second;
    }
}
