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
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.RandomCutForestState;

public class CompactSamplerMapperTest {
    private CompactSamplerMapper mapper;
    private RandomCutForestState forestState;

    private static int sampleSize = 20;
    private static double lambda = 0.01;

    private static class SamplerProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext extensionContext) throws Exception {
            int sampleSize = 20;
            double lambda = 0.01;
            long seed = 4444;
            CompactSampler sampler1 = new CompactSampler(sampleSize, lambda, seed, false);
            CompactSampler sampler2 = new CompactSampler(sampleSize, lambda, seed, true);

            Random random = new Random();
            long baseIndex = 10_000;
            for (int i = 0; i < 100; i++) {
                int pointReference = random.nextInt();
                sampler1.update(pointReference, baseIndex + i);
                sampler2.update(pointReference, baseIndex + i);
            }

            return Stream.of(sampler1, sampler2).map(Arguments::of);
        }
    }

    @BeforeEach
    public void setUp() {
        mapper = new CompactSamplerMapper();
        mapper.setValidateHeap(true);
        forestState = new RandomCutForestState();
        forestState.setSampleSize(sampleSize);
        forestState.setLambda(lambda);
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testRoundTrip(CompactSampler sampler) {
        CompactSampler sampler2 = mapper.toModel(mapper.toState(sampler), forestState);
        assertArrayEquals(sampler.getWeightArray(), sampler2.getWeightArray());
        assertArrayEquals(sampler.getPointIndexArray(), sampler2.getPointIndexArray());
        assertEquals(sampler.getCapacity(), sampler2.getCapacity());
        assertEquals(sampler.size(), sampler2.size());
        assertEquals(sampler.getLambda(), sampler2.getLambda());
        assertFalse(sampler2.getEvictedPoint().isPresent());

        if (sampler.isStoreSequenceIndexesEnabled()) {
            assertTrue(sampler2.isStoreSequenceIndexesEnabled());
            assertArrayEquals(sampler.getSequenceIndexArray(), sampler2.getSequenceIndexArray());
        } else {
            assertFalse(sampler2.isStoreSequenceIndexesEnabled());
            assertNull(sampler2.getSequenceIndexArray());
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testRoundTripInvalidHeap(CompactSampler sampler) {
        mapper.setValidateHeap(true);
        CompactSamplerState state = mapper.toState(sampler);

        // swap to weights in the weight array in order to violate the heap property
        float[] weights = state.getWeight();
        int index = weights.length / 4;
        float temp = weights[index];
        weights[index] = weights[2 * index + 1];
        weights[2 * index + 1] = temp;

        assertThrows(IllegalStateException.class, () -> mapper.toModel(state, forestState));

        mapper.setValidateHeap(false);
        CompactSampler sampler2 = mapper.toModel(state, forestState);
        assertArrayEquals(sampler.getWeightArray(), sampler2.getWeightArray());
    }
}
