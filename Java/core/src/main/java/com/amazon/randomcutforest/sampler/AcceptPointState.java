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

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * A container class used by {@link CompactSampler}. These sampler
 * implementations compute weights during {@link IStreamSampler#acceptPoint} to
 * determine if a new point should be added to the sample. This class retains
 * the sequence index and computed weight from that method call for use in the
 * subsequent {@link IStreamSampler#addPoint} call.
 */
@Data
@AllArgsConstructor
public class AcceptPointState {
    private long sequenceIndex;
    private float weight;
}
