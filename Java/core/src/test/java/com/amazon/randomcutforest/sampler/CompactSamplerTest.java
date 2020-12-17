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

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

public class CompactSamplerTest {

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
            return Stream.of(Arguments.of(random1, sampler1), Arguments.of(random2, sampler2));
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testNew(Random random, CompactSampler sampler) {
        // test CompactSampler fields not defined in the IStreamSampler interface
        assertEquals(lambda, sampler.getLambda());
        assertNotNull(sampler.getWeightArray());
        assertNotNull(sampler.getPointIndexArray());

        if (sampler.isStoreSequenceIndexesEnabled()) {
            assertNotNull(sampler.getSequenceIndexArray());
        } else {
            assertNull(sampler.getSequenceIndexArray());
        }
    }

    @Test
    public void testNewFromExistingWeights() {
        int sampleSize = 10;
        double lambda = 0.1;

        // weight array is valid heap
        float[] weight = { 0.4f, 0.3f, 0.2f };
        int[] pointIndex = { 1, 2, 3 };
        CompactSampler sampler = new CompactSampler(sampleSize, lambda, new Random(), weight, pointIndex, null, true);

        assertFalse(sampler.getEvictedPoint().isPresent());
        assertFalse(sampler.isStoreSequenceIndexesEnabled());
        assertEquals(3, sampler.size());
        assertNull(sampler.getSequenceIndexArray());

        for (int i = 0; i < 3; i++) {
            assertEquals(weight[i], sampler.weight[i]);
            assertEquals(pointIndex[i], sampler.pointIndex[i]);
        }
    }

    @Test
    public void testUniformSampler() {
        CompactSampler uniformSampler = CompactSampler.uniformSampler(sampleSize, seed, false);
        assertFalse(uniformSampler.getEvictedPoint().isPresent());
        assertFalse(uniformSampler.isReady());
        assertFalse(uniformSampler.isFull());
        assertEquals(sampleSize, uniformSampler.getCapacity());
        assertEquals(0, uniformSampler.size());
        assertEquals(0.0, uniformSampler.getLambda());
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testAddSample(Random random, CompactSampler sampler) {
        sampler.addSample(1, 0.5f, 10L);
        sampler.addSample(12, 1.2f, 11L);
        sampler.addSample(123, 0.03f, 12L);

        assertEquals(3, sampler.size());
        assertEquals(sampleSize, sampler.getCapacity());

        List<Weighted<Integer>> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(Weighted<Integer>::getWeight));
        assertEquals(3, samples.size());

        assertEquals(123, samples.get(0).getValue());
        assertEquals(0.03f, samples.get(0).getWeight());

        assertEquals(1, samples.get(1).getValue());
        assertEquals(0.5f, samples.get(1).getWeight());

        assertEquals(12, samples.get(2).getValue());
        assertEquals(1.2f, samples.get(2).getWeight());
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testAcceptSample(Random random, CompactSampler sampler) {
        // The sampler should accept all samples until the sampler is full
        for (int i = 0; i < sampleSize; i++) {
            Optional<Float> weight = sampler.acceptSample(i);
            assertTrue(weight.isPresent());
            sampler.addSample(i, weight.get(), i);
        }

        // In subsequent calls to sample, either the result is empty or else
        // the new weight is smaller than the evicted weight

        int numAccepted = 0;
        for (int i = sampleSize; i < 2 * sampleSize; i++) {
            Optional<Float> weight = sampler.acceptSample(i);
            if (weight.isPresent()) {
                numAccepted++;
                assertTrue(sampler.getEvictedPoint().isPresent());
                assertTrue(weight.get() < sampler.getEvictedPoint().get().getWeight());
                sampler.addSample(i, weight.get(), i);
            }
        }
        assertTrue(numAccepted > 0, "the sampler did not accept any points");
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testSample(Random random, CompactSampler compactSampler) {
        CompactSampler sampler = spy(compactSampler);
        for (int i = 0; i < sampleSize; i++) {
            assertTrue(sampler.sample(i, i));
        }

        // all points should be added to the sampler until the sampler is full
        assertEquals(sampleSize, sampler.size());
        verify(sampler, times(sampleSize)).addSample(any(), anyFloat(), anyLong());

        reset(sampler);

        int numSampled = 0;
        for (int i = sampleSize; i < 2 * sampleSize; i++) {
            if (sampler.sample(i, i)) {
                numSampled++;
            }
        }
        assertTrue(numSampled > 0, "no new values were sampled");
        assertTrue(numSampled < sampleSize, "all values were sampled");

        verify(sampler, times(numSampled)).addSample(any(), anyFloat(), anyLong());
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testGetScore(Random random, CompactSampler sampler) {
        when(random.nextDouble()).thenReturn(0.25).thenReturn(0.75).thenReturn(0.50);

        sampler.sample(1, 101);
        sampler.sample(2, 102);
        sampler.sample(3, 103);

        double[] expectedScores = new double[3];
        expectedScores[0] = -lambda * 101L + Math.log(-Math.log(0.25));
        expectedScores[1] = -lambda * 102L + Math.log(-Math.log(0.75));
        expectedScores[2] = -lambda * 103L + Math.log(-Math.log(0.50));
        Arrays.sort(expectedScores);

        List<Weighted<Integer>> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(Weighted<Integer>::getWeight));

        for (int i = 0; i < 3; i++) {
            assertEquals(expectedScores[i], samples.get(i).getWeight(), EPSILON);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SamplerProvider.class)
    public void testValidateHeap(Random random, CompactSampler sampler) {
        // populate the heap
        for (int i = 0; i < 2 * sampleSize; i++) {
            sampler.sample(i, i);
        }

        float[] weightArray = sampler.getWeightArray();

        // swapping a weight value with one of its children will break the heap property
        int i = sampleSize / 4;
        float f = weightArray[i];
        weightArray[i] = weightArray[2 * i + 1];
        weightArray[2 * i + 1] = f;

        assertThrows(IllegalStateException.class, () -> new CompactSampler(sampleSize, lambda, random, weightArray,
                sampler.getPointIndexArray(), sampler.getSequenceIndexArray(), true));
    }
}
