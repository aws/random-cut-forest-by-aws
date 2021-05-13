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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.sampler.CompactSampler;

public class CompactSamplerMapperTest {

    private static int sampleSize = 20;
    private static double lambda = 0.01;
    private static long seed = 4444;

    public static Stream<Arguments> nonemptySamplerProvider() {
        CompactSampler fullSampler1 = new CompactSampler(sampleSize, lambda, seed, false);
        CompactSampler fullSampler2 = new CompactSampler(sampleSize, lambda, seed, true);

        Random random = new Random();
        long baseIndex = 10_000;
        for (int i = 0; i < 100; i++) {
            int pointReference = random.nextInt();
            fullSampler1.update(pointReference, baseIndex + i);
            fullSampler2.update(pointReference, baseIndex + i);
        }

        CompactSampler partiallyFullSampler1 = new CompactSampler(sampleSize, lambda, seed, false);
        CompactSampler partiallyFullSampler2 = new CompactSampler(sampleSize, lambda, seed, true);

        for (int i = 0; i < sampleSize / 2; i++) {
            int pointReference = random.nextInt();
            partiallyFullSampler1.update(pointReference, baseIndex + i);
            partiallyFullSampler2.update(pointReference, baseIndex + i);
        }

        return Stream.of(Arguments.of("full sampler without sequence indexes", fullSampler1),
                Arguments.of("full sampler with sequence indexes", fullSampler2),
                Arguments.of("partially full sampler without sequence indexes", partiallyFullSampler1),
                Arguments.of("partially full sampler with sequence indexes", partiallyFullSampler2));
    }

    public static Stream<Arguments> samplerProvider() {
        CompactSampler emptySampler1 = new CompactSampler(sampleSize, lambda, seed, false);
        CompactSampler emptySampler2 = new CompactSampler(sampleSize, lambda, seed, true);

        return Stream.concat(nonemptySamplerProvider(),
                Stream.of(Arguments.of("empty sampler without sequence indexes", emptySampler1),
                        Arguments.of("empty sampler with sequence indexes", emptySampler2)));
    }

    private CompactSamplerMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new CompactSamplerMapper();
        mapper.setValidateHeapEnabled(false);
    }

    private void assertValidMapping(CompactSampler original, CompactSampler mapped) {
        assertArrayEquals(original.getWeightArray(), mapped.getWeightArray(), "different weight arrays");
        assertArrayEquals(original.getPointIndexArray(), mapped.getPointIndexArray(), "different point index arrays");
        assertEquals(original.getCapacity(), mapped.getCapacity());
        assertEquals(original.size(), mapped.size());
        assertEquals(original.getTimeDecay(), mapped.getTimeDecay());
        assertFalse(mapped.getEvictedPoint().isPresent());

        if (original.isStoreSequenceIndexesEnabled()) {
            assertTrue(mapped.isStoreSequenceIndexesEnabled());
            assertArrayEquals(original.getSequenceIndexArray(), mapped.getSequenceIndexArray(),
                    "different sequence index arrays");
        } else {
            assertFalse(mapped.isStoreSequenceIndexesEnabled());
            assertNull(mapped.getSequenceIndexArray());
        }
    }

    @ParameterizedTest
    @MethodSource("nonemptySamplerProvider")
    public void testRoundTripInvalidHeap(String description, CompactSampler sampler) {
        mapper.setValidateHeapEnabled(true);
        CompactSamplerState state = mapper.toState(sampler);

        // swap to weights in the weight array in order to violate the heap property
        float[] weights = state.getWeight();
        int index = state.getSize() / 4;
        float temp = weights[index];
        weights[index] = weights[2 * index + 1];
        weights[2 * index + 1] = temp;

        assertThrows(IllegalStateException.class, () -> mapper.toModel(state));

        mapper.setValidateHeapEnabled(false);
        CompactSampler sampler2 = mapper.toModel(state);
        assertArrayEquals(sampler.getWeightArray(), sampler2.getWeightArray());
    }
}
