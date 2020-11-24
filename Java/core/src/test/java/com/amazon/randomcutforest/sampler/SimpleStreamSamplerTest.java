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
import static org.junit.jupiter.api.Assertions.*;
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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class SimpleStreamSamplerTest {

    private int sampleSize;
    private double lambda;
    private long seed;
    private Random random;
    private SimpleStreamSampler<double[]> sampler;

    @BeforeEach
    public void setUp() {
        sampleSize = 101;
        lambda = 0.01;
        seed = 42L;
        random = spy(new Random(seed));
        sampler = new SimpleStreamSampler<>(sampleSize, lambda, random, false);
    }

    @Test
    public void testNew() {
        // test fields defined in SimpleStreamSampler that aren't part of the
        // IStreamSampler interface
        assertEquals(lambda, sampler.getLambda());

        SimpleStreamSampler<double[]> uniformSampler = new SimpleStreamSampler<>(11, 0, 14, false);
        assertFalse(uniformSampler.getEvictedPoint().isPresent());
        assertFalse(uniformSampler.isReady());
        assertFalse(uniformSampler.isFull());
        assertEquals(11, uniformSampler.getCapacity());
        assertEquals(0, uniformSampler.size());
        assertEquals(0.0, uniformSampler.getLambda());
    }

    @Test
    public void testAddSample() {
        assertEquals(0, sampler.size());
        assertEquals(sampleSize, sampler.getCapacity());

        sampler.addSample(new double[] { 1.1 }, 0.5f, 10L);
        sampler.addSample(new double[] { -2.2 }, 1.2f, 11L);
        sampler.addSample(new double[] { 3.3 }, 0.03f, 12L);

        assertEquals(3, sampler.size());
        assertEquals(sampleSize, sampler.getCapacity());

        List<Weighted<double[]>> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(Weighted<double[]>::getWeight));
        assertEquals(3, samples.size());

        assertArrayEquals(new double[] { 3.3 }, samples.get(0).getValue());
        assertEquals(0.03f, samples.get(0).getWeight());

        assertArrayEquals(new double[] { 1.1 }, samples.get(1).getValue());
        assertEquals(0.5f, samples.get(1).getWeight());

        assertArrayEquals(new double[] { -2.2 }, samples.get(2).getValue());
        assertEquals(1.2f, samples.get(2).getWeight());
    }

    @Test
    public void testAcceptSample() {
        // The sampler should accept all samples until the sampler is full
        for (int i = 0; i < sampleSize; i++) {
            Optional<Float> weight = sampler.acceptSample(i);
            assertTrue(weight.isPresent());
            sampler.addSample(new double[] { Math.random() }, weight.get(), i);
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
                sampler.addSample(new double[] { Math.random() }, weight.get(), i);
            }
        }
        assertTrue(numAccepted > 0, "the sampler did not accept any points");
    }

    @Test
    public void testSample() {
        SimpleStreamSampler<double[]> samplerSpy = spy(sampler);
        for (int i = 0; i < sampleSize; i++) {
            assertTrue(samplerSpy.sample(new double[] { i + 0.0 }, i));
        }

        // all points should be added to the sampler until the sampler is full
        assertEquals(sampleSize, samplerSpy.size());
        verify(samplerSpy, times(sampleSize)).addSample(any(), anyFloat(), anyLong());

        reset(samplerSpy);

        int numSampled = 0;
        for (int i = sampleSize; i < 2 * sampleSize; i++) {
            if (samplerSpy.sample(new double[] { i + 0.0 }, i)) {
                numSampled++;
            }
        }
        assertTrue(numSampled > 0, "no new values were sampled");
        assertTrue(numSampled < sampleSize, "all values were sampled");

        verify(samplerSpy, times(numSampled)).addSample(any(), anyFloat(), anyLong());
    }

    @Test
    public void testGetScore() {
        when(random.nextDouble()).thenReturn(0.25).thenReturn(0.75).thenReturn(0.50);

        sampler.sample(new double[] { 1.0 }, 101);
        sampler.sample(new double[] { 2.0 }, 102);
        sampler.sample(new double[] { 3.0 }, 103);

        double[] expectedScores = new double[3];
        expectedScores[0] = -lambda * 101L + Math.log(-Math.log(0.25));
        expectedScores[1] = -lambda * 102L + Math.log(-Math.log(0.75));
        expectedScores[2] = -lambda * 103L + Math.log(-Math.log(0.50));
        Arrays.sort(expectedScores);

        List<Weighted<double[]>> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(Weighted<double[]>::getWeight));

        for (int i = 0; i < 3; i++) {
            assertEquals(expectedScores[i], samples.get(i).getWeight(), EPSILON);
        }
    }
}
