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

package com.amazon.randomcutforest.parkservices.config;

/**
 * Options for using RCF, specially with thresholds
 */
public enum ScoringStrategy {

    /**
     * default behavior to be optimized; currently EXPECTED_INVERSE_DEPTH
     */
    EXPECTED_INVERSE_DEPTH,

    /**
     * This is the same as STANDARD mode where the scoring function is switched to
     * distances between the vectors. Since RCFs build a multiresolution tree, and
     * in the aggregate, preserves distances to some approximation, this provides an
     * alternate anomaly detection mechanism which can be useful for shingleSize = 1
     * and (dynamic) population analysis via RCFs. Specifically it switches the
     * scoring to be based on the distance computation in the Density Estimation
     * (interpolation). This allows for a direct comparison of clustering based
     * outlier detection and RCFs over numeric vectors. All transformations
     * available to the STANDARD mode in the ThresholdedRCF are available for this
     * mode as well; this does not affect RandomCutForest core in any way. For
     * timeseries analysis the STANDARD mode is recommended, but this does provide
     * another option in combination with the TransformMethods.
     */
    DISTANCE,

    /**
     * RCFs are an updatable data structure that can support multiple difference
     * inference methods. Given the longstanding interest in ensembles of different
     * models, this strategy uses the multiple inference capabilities to increase
     * precision. It does not escape our attention that multi-mode allows the
     * functionality of multi-models yet use a significantly smaller state/memory
     * footprint since all the modes use RCF. The different modes are probed with
     * computational efficiency in mind.
     */

    MULTI_MODE,

    /**
     * Same as above, except optimized for increasing recall.
     */

    MULTI_MODE_RECALL;

}
