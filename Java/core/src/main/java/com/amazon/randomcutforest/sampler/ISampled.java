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

/**
 * A simple wrapper class representing a point that has been sampled by a
 * sampler. A sampled point can be added to or removed from a tree.
 * 
 * @param <P> The point representation used by this sampled point.
 */
public interface ISampled<P> {
    /**
     * Return the sampled value.
     * 
     * @return the sampled value.
     */
    P getValue();

    /**
     * Return the sequence index of the sampled value.
     * 
     * @return the sequence index of the sampled value.
     */
    long getSequenceIndex();
}
