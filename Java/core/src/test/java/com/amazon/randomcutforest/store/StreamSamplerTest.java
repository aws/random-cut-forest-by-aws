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

package com.amazon.randomcutforest.store;

import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class StreamSamplerTest {

    @Test
    void testBuilder() {
        StreamSampler.Builder builder = StreamSampler.builder().capacity(10).timeDecay(0).randomSeed(0);
        assertTrue(builder.getCapacity() == 10);
        assertTrue(builder.getRandomSeed() == 0);
        assertTrue(builder.getTimeDecay() == 0);
    }

    @Test
    void testConstructor() {
        StreamSampler<float[]> sampler = StreamSampler.builder().initialAcceptFraction(1.0)
                .storeSequenceIndexesEnabled(true).build();
        assertEquals(sampler.getEntriesSeen(), 0);
        assertEquals(sampler.getSequenceNumber(), -1L);
        sampler.sample(new float[] {}, 1f);
        StreamSampler<float[]> second = StreamSampler.builder().initialAcceptFraction(0.5)
                .storeSequenceIndexesEnabled(false).build();
        second.sample(new float[] {}, 0.5f);
        second.sample(new float[] {}, 2f);
        assertThrows(IllegalArgumentException.class, () -> new StreamSampler(sampler, second, 0, 0, 0L));
        StreamSampler<float[]> merged = new StreamSampler(sampler, second, 10, 0, 0L);
        assertEquals(merged.entriesSeen, 3);
        assertEquals(merged.sampler.getInitialAcceptFraction(), 1.0);
        assertEquals(merged.getSequenceNumber(), 1);
    }

    @Test
    public void testConfig() {
        StreamSampler<float[]> sampler = StreamSampler.builder().initialAcceptFraction(1.0).build();
        assertTrue(sampler.isCurrentlySampling());
        assertTrue(sampler.getEntriesSeen() == 0);
        sampler.pauseSampling();
        assertFalse(sampler.isCurrentlySampling());
        sampler.sample(new float[] {}, 0.1f);
        assertTrue(sampler.getEntriesSeen() == 1);
        assertTrue(sampler.getObjectList().size() == 0);
        sampler.resumeSampling();
        assertTrue(sampler.isCurrentlySampling());
        sampler.sample(new float[] { 1.0f, 1.0f }, 0.2f);
        assertTrue(sampler.getEntriesSeen() == 2);
        assertTrue(sampler.getObjectList().size() == 1);
        sampler.pauseSampling();
        assertFalse(sampler.isCurrentlySampling());
        sampler.sample(new float[] { 1.0f, 1.0f }, 0.2f);
        assertTrue(sampler.getEntriesSeen() == 3);
        assertTrue(sampler.getObjectList().size() == 1);
        assertEquals(sampler.getCapacity(), DEFAULT_SAMPLE_SIZE);
        sampler.resumeSampling();
        for (int i = 0; i < 10000; i++) {
            sampler.sample(new float[] { 1.0f, 1.0f }, 1f);
        }
    }

}
