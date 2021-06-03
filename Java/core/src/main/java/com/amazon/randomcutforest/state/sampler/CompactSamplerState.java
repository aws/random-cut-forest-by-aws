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

package com.amazon.randomcutforest.state.sampler;

import static com.amazon.randomcutforest.state.Version.V2_0;

import lombok.Data;

/**
 * A data object representing the state of a
 * {@link com.amazon.randomcutforest.sampler.CompactSampler}.
 */
@Data
public class CompactSamplerState {
    /**
     * a version string for extensibility
     */
    private String version = V2_0;

    /**
     * An array of sampler weights.
     */
    private float[] weight;
    /**
     * An array of index values identifying the points in the sample. These indexes
     * will correspond to a {@link com.amazon.randomcutforest.store.PointStore}.
     */
    private int[] pointIndex;
    /**
     * boolean for deciding to store sequence indices
     */
    private boolean storeSequenceIndicesEnabled;
    /**
     * The sequence indexes of points in the sample.
     */
    private long[] sequenceIndex;
    /**
     * The number of points in the sample.
     */
    private int size;
    /**
     * The maximum number of points that the sampler can contain.
     */
    private int capacity;
    /**
     * The behavior of the sampler at initial sampling
     */
    private double initialAcceptFraction;
    /**
     * The time-decay parameter for this sampler
     */
    private double timeDecay;
    /**
     * Last update of lambda
     */
    private long sequenceIndexOfMostRecentTimeDecayUpdate;
    /**
     * maximum timestamp seen in update/computeWeight
     */
    private long maxSequenceIndex;
    /**
     * boolean indicating if the compression is enabled
     */
    private boolean compressed;
    /**
     * saving the random state, if desired
     */
    private long randomSeed;
}
