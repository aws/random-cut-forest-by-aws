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
 * Options for internally transforming data in RCF These are built for
 * convenience. Domain knowledge before feeding data into RCF(any tool) will
 * often have the best benefit! These apply to the basic data and not
 * timestamps, time is (hopefully) always moving forward and is measured shifted
 * (from a running mean), with an option of normalization.
 */
public enum TransformMethod {

    /**
     * the best transformation for data!
     */
    NONE,
    /**
     * standard column normalization using fixed weights
     */
    WEIGHTED,
    /**
     * subtract a moving average -- the average would be computed using the same
     * discount factor as the time decay of the RCF samplers.
     */
    SUBTRACT_MA,
    /**
     * divide by standard deviation, after subtracting MA
     */
    NORMALIZE,
    /**
     * difference from previous
     */
    DIFFERENCE,
    /**
     * divide by standard deviation of difference, after differencing (again
     * subtract MA)
     */
    NORMALIZE_DIFFERENCE;
}
