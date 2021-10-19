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
 * Options for filling in missing values
 */
public enum ImputationMethod {

    /**
     * use all 0's
     */
    ZERO,
    /**
     * use a fixed set of specified values (same as input dimension)
     */
    FIXED_VALUES,
    /**
     * last known value in each input dimension
     */
    PREVIOUS,
    /**
     * next seen value in each input dimension
     */
    NEXT,
    /**
     * linear interpolation
     */
    LINEAR,
    /**
     * use the RCF imputation; but would often require a minimum number of
     * observations and currently defaults to LINEAR
     */
    RCF;
}
