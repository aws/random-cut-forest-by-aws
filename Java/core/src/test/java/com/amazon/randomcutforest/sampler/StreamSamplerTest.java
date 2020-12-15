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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;

import java.util.Optional;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

/**
 * Test common functionality for IStreamSampler implementations
 */
public class StreamSamplerTest {
    private static int sampleSize = 101;
    private static double lambda = 0.01;
    private static long seed = 42L;

    private static class SamplerProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {
            Random random1 = spy(new Random(seed));
            CompactSampler sampler1 = new CompactSampler(sampleSize, lambda, random1, false);

            Random random2 = spy(new Random(seed));
            CompactSampler sampler2 = new CompactSampler(sampleSize, lambda, random2, true);

            Random random3 = spy(new Random(seed));
            SimpleStreamSampler<Integer> sampler3 = new SimpleStreamSampler<>(sampleSize, lambda, random3, false);

            Random random4 = spy(new Random(seed));
            SimpleStreamSampler<Integer> sampler4 = new SimpleStreamSampler<>(sampleSize, lambda, random4, true);

            return Stream.of(Arguments.of(random1, sampler1), Arguments.of(random2, sampler2),
                    Arguments.of(random3, sampler3), Arguments.of(random4, sampler4));
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testNew(Random random, IStreamSampler<?> sampler) {
        assertFalse(sampler.getEvictedPoint().isPresent());
        assertFalse(sampler.isReady());
        assertFalse(sampler.isFull());
        assertEquals(sampleSize, sampler.getCapacity());
        assertEquals(0, sampler.size());
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testIsReadyIsFull(Random random, IStreamSampler<Integer> sampler) {
        int i;
        for (i = 1; i < sampleSize / 4; i++) {
            assertTrue(sampler.update((int) Math.ceil(Math.random() * 100), i));
            assertFalse(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.size());
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        for (i = sampleSize / 4; i < sampleSize; i++) {
            assertTrue(sampler.update((int) Math.ceil(Math.random() * 100), i));
            assertTrue(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.size());
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        assertTrue(sampler.update((int) Math.ceil(Math.random() * 100), sampleSize));
        assertTrue(sampler.isReady());
        assertTrue(sampler.isFull());
        assertEquals(i, sampler.size());
        assertFalse(sampler.getEvictedPoint().isPresent());

        Optional<ISampled<double[]>> evicted;
        for (i = sampleSize + 1; i < 2 * sampleSize; i++) {
            // Either the sampling and the evicted point are both null or both non-null
            assertTrue(
                    sampler.update((int) Math.ceil(Math.random() * 100), i) == sampler.getEvictedPoint().isPresent());

            assertTrue(sampler.isReady());
            assertTrue(sampler.isFull());
            assertEquals(sampleSize, sampler.size());

        }
    }

    /**
     * This test shows that SimpleStreamSampler and CompactSampler have the same
     * sampling behavior.
     */
    @Test
    public void testCompareSampling() {
        CompactSampler compact = new CompactSampler(10, 0.001, 10, false);
        SimpleStreamSampler<double[]> simple = new SimpleStreamSampler<>(10, 0.001, 10, false);

        for (int i = 0; i < 100000; i++) {
            boolean accepted1 = compact.acceptPoint(i);
            boolean accepted2 = simple.acceptPoint(i);
            assertEquals(accepted1, accepted2);
            if (accepted1) {
                assertEquals(compact.acceptPointState.getWeight(), simple.acceptPointState.getWeight(), 1E-10);
                compact.addPoint(i);
                simple.addPoint(new double[] { i });
            }
            assertEquals(compact.getEvictedPoint().isPresent(), simple.getEvictedPoint().isPresent());
            if (compact.getEvictedPoint().isPresent()) {
                Weighted<Integer> evictedCompact = (Weighted<Integer>) compact.getEvictedPoint().get();
                Weighted<double[]> evictedSimple = (Weighted<double[]>) simple.getEvictedPoint().get();
                int y = evictedCompact.getValue();
                double[] x = evictedSimple.getValue();
                assertEquals(x[0], y);
                assertEquals(evictedCompact.getWeight(), evictedSimple.getWeight(), 1E-10);
            }
        }
    }
}
