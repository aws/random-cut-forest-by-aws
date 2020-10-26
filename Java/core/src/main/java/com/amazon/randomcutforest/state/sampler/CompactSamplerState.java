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

import java.util.Arrays;

import lombok.Data;

import com.amazon.randomcutforest.sampler.CompactSampler;

@Data
public class CompactSamplerState {

    public float[] weightArray;

    public int[] referenceArray;

    public long[] sequenceArray;

    public int size;

    public int capacity;

    public CompactSamplerState() {

    }

    public CompactSamplerState(CompactSampler sampler) {
        size = sampler.size();
        capacity = sampler.getCapacity();
        weightArray = Arrays.copyOf(sampler.getWeightArray(), size);
        referenceArray = Arrays.copyOf(sampler.getReferenceArray(), size);
        if (sampler.getSequenceArray() != null) {
            sequenceArray = Arrays.copyOf(sampler.getSequenceArray(), size);
        } else {
            sequenceArray = null;
        }
    }

    public CompactSamplerState(int size, int capacity, boolean storeSeq) {
        this.size = size;
        this.capacity = capacity;
        weightArray = new float[size];
        referenceArray = new int[size];
        if (storeSeq) {
            sequenceArray = new long[size];
        } else {
            sequenceArray = null;
        }
    }
}
