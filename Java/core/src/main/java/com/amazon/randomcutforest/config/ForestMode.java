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

package com.amazon.randomcutforest.config;

/**
 * Options for using RCF, specially with thresholds
 */
public enum ForestMode {

    /**
     * a standard mode that uses shingling and most known applications; it uses the
     * last K data points where K=1 would correspond to non time series (population)
     * analysis
     */
    STANDARD,
    /**
     * time stamp is added automatically to data to correlate within RCF itself;
     * this is useful for event streaams and for modeling sparse events. Option is
     * provided to normalize the time gaps.
     */
    TIME_AUGMENTED,
    /**
     * uses various Fill-In strageies for data with gaps but not really sparse. Must
     * have shingleSize greater than 1, typically larger shingle size is better, and
     * so is fewer input dimensions
     */
    STREAMING_IMPUTE,
    /**
     * This is the same as STANDARD mode where the scoring function is switched to
     * distances between the vectors. Since RCFs build a multiresolution tree, and
     * in the aggregate, preserves distances to some approximation, this provides an
     * alternate anomaly detection mechanism which can be useful for shingleSize = 1
     * and (dynamic) population analysis via RCFs. Specifially it switches the
     * scoring to be based on the distance computation in the Density Estimation
     * (interpolation). This allows for a direct comparison of clustering based
     * outlier detection and RCFs over numeric vectors. All transformations
     * available to the STANDARD mode in the ThresholdedRCF are available for this
     * mode as well; this does not affect RandomCutForest core in any way. For
     * timeseries analysis the STANDARD mode is recommended, but this does provide
     * another option in combination with the TransformMethods.
     */
    DISTANCE;

}
