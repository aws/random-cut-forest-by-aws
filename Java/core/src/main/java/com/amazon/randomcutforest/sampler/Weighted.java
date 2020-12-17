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

import lombok.Data;

/**
 * A container class representing a weighted value. This generic type is used in
 * sampler implementations to store points along with weights that were computed
 * as part of sampling.
 *
 * @param <P> The representation of the point value.
 */
@Data
public class Weighted<P> {

    /**
     * The contained value.
     */
    private final P value;

    /**
     * The weight assigned to this value.
     */
    private final float weight;
}
