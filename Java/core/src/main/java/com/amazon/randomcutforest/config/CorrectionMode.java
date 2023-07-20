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
public enum CorrectionMode {

    /**
     * default behavior, no correction
     */
    NONE,

    /**
     * due to transforms, or due to input noise
     */
    NOISE,

    /**
     * elimination due to multi mode operation
     */

    MULTI_MODE,

    /**
     * effect of an anomaly in shingle
     */

    ANOMALY_IN_SHINGLE,

    /**
     * conditional forecast, using conditional fields
     */

    CONDITIONAL_FORECAST,

    /**
     * forecasted value was not very different
     */

    FORECAST,

    /**
     * data drifts and level shifts, will not be corrected unless level shifts are
     * turned on
     */

    DATA_DRIFT

}
