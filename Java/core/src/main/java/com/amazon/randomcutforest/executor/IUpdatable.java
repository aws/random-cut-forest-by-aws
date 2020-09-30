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

import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.sampler.Weighted;

public interface IUpdatable<P> {
    /**
     * result of an update on a sampler plus tree
     * 
     * @param point  to be considered for updating the sampler plus tree
     * @param seqNum timestamp
     * @return the (inserted,deleted) pair of handles in the tree for eventual
     *         bookkeeping
     */
    Optional<UpdateReturn<P>> update(P point, long seqNum);

    /**
     * returns the sampler's queue
     * 
     * @return the list of weighted samples, without the sequence information
     */
    List<Weighted<P>> getWeightedSamples();

    /**
     *
     * @return the list of sequential samples
     */
    List<Sequential<P>> getSequentialSamples();

    /**
     * initialize the models
     * 
     * @param samples    samples without sequence information
     * @param seqSamples samples with sequence information exactly one of these
     *                   should be non-null
     */

    void initialize(List<Weighted<P>> samples, List<Sequential<P>> seqSamples);

}
