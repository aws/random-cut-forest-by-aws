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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.Sequential;

public class SimpleStreamSamplerV2Test {

    private int sampleSize;
    private double lambda;
    private long seed;
    private Random random;
    private SimpleStreamSamplerV2<double[]> sampler;

    @BeforeEach
    public void setUp() {
        sampleSize = 101;
        lambda = 0.01;
        seed = 42L;
        random = spy(new Random(seed));
        sampler = new SimpleStreamSamplerV2<>(double[].class, sampleSize, lambda, random);
    }

    @Test
    public void testNew() {
        assertFalse(sampler.getEvictedPoint().isPresent());
        assertFalse(sampler.isReady());
        assertFalse(sampler.isFull());
        assertEquals(sampleSize, sampler.getCapacity());
        assertEquals(0, sampler.getSize());
        assertEquals(lambda, sampler.getLambda());

        SimpleStreamSamplerV2<double[]> uniformSampler = SimpleStreamSamplerV2.uniformSampler(double[].class, 11, 14);
        assertFalse(uniformSampler.getEvictedPoint().isPresent());
        assertFalse(uniformSampler.isReady());
        assertFalse(uniformSampler.isFull());
        assertEquals(11, uniformSampler.getCapacity());
        assertEquals(0, uniformSampler.getSize());
        assertEquals(0.0, uniformSampler.getLambda());
    }

    @Test
    public void testPointComparator() {
        Weighted<double[]> point1 = new Weighted<>(new double[] { 0.99, -55.2 }, 999L, 1.23);
        Weighted<double[]> point2 = new Weighted<>(new double[] { 2.2, 87.0 }, 1000L, -77);
        Weighted<double[]> point3 = new Weighted<>(new double[] { -2.1, 99.4 }, 1001L, -77);

        Comparator<Weighted<double[]>> comparator = SimpleStreamSamplerV2.getComparator(double[].class);
        assertTrue(comparator.compare(point1, point2) < 0);
        assertTrue(comparator.compare(point3, point1) > 0);
        assertTrue(comparator.compare(point2, point3) == 0);
    }

    @Test
    public void testGetScore() {
        when(random.nextDouble()).thenReturn(0.25).thenReturn(0.75).thenReturn(0.50);

        sampler.sample(new Sequential<>(new double[] { -0.1 }, 101));
        sampler.sample(new Sequential<>(new double[] { 11.1 }, 102));
        sampler.sample(new Sequential<>(new double[] { 99.8 }, 103));

        double[] expectedScores = new double[3];
        expectedScores[0] = -lambda * 101L + Math.log(-Math.log(0.25));
        expectedScores[1] = -lambda * 102L + Math.log(-Math.log(0.75));
        expectedScores[2] = -lambda * 103L + Math.log(-Math.log(0.50));
        Arrays.sort(expectedScores);

        List<Weighted<double[]>> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(Weighted::getWeight));

        for (int i = 0; i < 3; i++) {
            assertEquals(expectedScores[i], samples.get(i).getWeight(), EPSILON);
        }
    }

    @Test
    public void testIsReadyIsFull() {
        boolean sampled;
        int i;
        for (i = 1; i < sampleSize / 4; i++) {
            sampled = sampler.sample(new Sequential<>(new double[] { Math.random() }, i));
            assertFalse(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertTrue(sampled);
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        for (i = sampleSize / 4; i < sampleSize; i++) {
            sampled = sampler.sample(new Sequential<>(new double[] { Math.random() }, i));
            assertTrue(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertTrue(sampled);
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        sampled = sampler.sample(new Sequential<>(new double[] { Math.random() }, sampleSize));
        assertTrue(sampler.isReady());
        assertTrue(sampler.isFull());
        assertEquals(i, sampler.getSize());
        assertTrue(sampled);
        assertFalse(sampler.getEvictedPoint().isPresent());

        Optional<Sequential<double[]>> evictedPoint;

        for (i = sampleSize + 1; i < 2 * sampleSize; i++) {
            sampled = sampler.sample(new Sequential<>(new double[] { Math.random() }, i));
            evictedPoint = sampler.getEvictedPoint();
            assertTrue(sampler.isReady());
            assertTrue(sampler.isFull());
            assertEquals(sampleSize, sampler.getSize());

            // Either the the point was sampled and the evicted point is preset, or the
            // point was not sampled and the evicted point is empty
            assertEquals(sampled, evictedPoint.isPresent());
        }
    }

    @Test
    public void testSample() {
        // first populate the sampler:

        int entriesSeen;
        for (entriesSeen = 0; entriesSeen < 2 * sampleSize; entriesSeen++) {
            sampler.sample(new Sequential<>(new double[] { Math.random() }, entriesSeen));
        }

        assertEquals(entriesSeen, 2 * sampleSize);
        assertTrue(sampler.isFull());

        // find the lowest weight currently in the sampler
        double maxWeight = sampler.getWeightedSamples().stream().mapToDouble(Weighted::getWeight).max()
                .orElseThrow(IllegalStateException::new);

        // First choose a random value U so that
        // -lambda * entriesSeen + log(-log(U)) > maxWeight
        // which is equivalent to
        // U < exp(-exp(maxWeight + lambda * entriesSeen))
        // using this formula results in an underflow, so just use a very small number

        double u = 10e-100;
        when(random.nextDouble()).thenReturn(u);

        // With this choice of u, the next sample should be rejected
        boolean sampled = sampler.sample(new Sequential<>(new double[] { Math.random() }, ++entriesSeen));
        assertFalse(sampled);
        assertFalse(sampler.getEvictedPoint().isPresent());

        double maxWeight2 = sampler.getWeightedSamples().stream().mapToDouble(Weighted::getWeight).max()
                .orElseThrow(IllegalStateException::new);

        assertEquals(maxWeight, maxWeight2);

        // Next choose a large value of u (i.e., close to 1)
        // For this choice of U, the new point should be accepted

        u = 1 - 10e-100;
        when(random.nextDouble()).thenReturn(u);

        double point = Math.random();
        sampled = sampler.sample(new Sequential<>(new double[] { point }, ++entriesSeen));

        assertTrue(sampled);
        assertNotNull(sampler.getEvictedPoint());

        maxWeight2 = sampler.getWeightedSamples().stream().mapToDouble(Weighted::getWeight).max()
                .orElseThrow(IllegalStateException::new);
        assertTrue(maxWeight2 < maxWeight);
        assertTrue(sampler.getSamples().stream().anyMatch(p -> p.getValue()[0] == point));
    }
}
