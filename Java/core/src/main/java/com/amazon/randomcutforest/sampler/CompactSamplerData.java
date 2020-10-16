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

import java.util.Arrays;

public class CompactSamplerData {

    public final float[] weightArray;

    public final int[] referenceArray;

    public final long[] sequenceArray;

    public final int currentSize;

    public final int maxSize;

    public CompactSamplerData(CompactSampler sampler) {
        currentSize = sampler.currentSize;
        maxSize = sampler.maxSize;
        weightArray = Arrays.copyOf(sampler.weightArray, currentSize);
        referenceArray = Arrays.copyOf(sampler.referenceArray, currentSize);
        if (sampler.sequenceArray != null) {
            sequenceArray = Arrays.copyOf(sampler.sequenceArray, currentSize);
        } else {
            sequenceArray = null;
        }
    }

    public CompactSamplerData(int currentSize, int maxSize, boolean storeSeq) {
        this.currentSize = currentSize;
        this.maxSize = maxSize;
        weightArray = new float[currentSize];
        referenceArray = new int[currentSize];
        if (storeSeq) {
            sequenceArray = new long[currentSize];
        } else {
            sequenceArray = null;
        }
    }
}
