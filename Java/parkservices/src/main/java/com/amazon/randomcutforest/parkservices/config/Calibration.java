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

public enum Calibration {

    NONE,

    /**
     * a basic staring point where the intervals are adjusted to be the minimal
     * necessary based on past error the intervals are smaller -- but the interval
     * precision will likely be close to 1 - 2 * percentile
     */
    MINIMAL,

    /**
     * a Markov inequality based interval, where the past error and model errors are
     * additive. The interval precision is likely higher than MINIMAL but so are the
     * intervals.
     */
    SIMPLE;

}
