/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class SimpleStreamSamplerTest {

    private int sampleSize;
    private double lambda;
    private long seed;
    private Random random;
    private SimpleStreamSampler sampler;

    @BeforeEach
    public void setUp() {
        sampleSize = 101;
        lambda = 0.01;
        seed = 42L;
        random = spy(new Random(seed));
        sampler = new SimpleStreamSampler(sampleSize, lambda, random);
    }

    @Test
    public void testNew() {
        assertNull(sampler.getEvictedPoint());
        assertFalse(sampler.isReady());
        assertFalse(sampler.isFull());
        assertEquals(sampleSize, sampler.getCapacity());
        assertEquals(0, sampler.getSize());
        assertEquals(lambda, sampler.getLambda());

        SimpleStreamSampler uniformSampler = SimpleStreamSampler.uniformSampler(11, 14);
        assertNull(uniformSampler.getEvictedPoint());
        assertFalse(uniformSampler.isReady());
        assertFalse(uniformSampler.isFull());
        assertEquals(11, uniformSampler.getCapacity());
        assertEquals(0, uniformSampler.getSize());
        assertEquals(0.0, uniformSampler.getLambda());
    }

    @Test
    public void testPointComparator() {
        WeightedPoint point1 = new WeightedPoint(new double[] { 0.99, -55.2 }, 999L, 1.23);
        WeightedPoint point2 = new WeightedPoint(new double[] { 2.2, 87.0 }, 1000L, -77);
        WeightedPoint point3 = new WeightedPoint(new double[] { -2.1, 99.4 }, 1001L, -77);

        assertTrue(SimpleStreamSampler.POINT_COMPARATOR.compare(point1, point2) < 0);
        assertTrue(SimpleStreamSampler.POINT_COMPARATOR.compare(point3, point1) > 0);
        assertTrue(SimpleStreamSampler.POINT_COMPARATOR.compare(point2, point3) == 0);
    }

    @Test
    public void testGetScore() {
        when(random.nextDouble()).thenReturn(0.25).thenReturn(0.75).thenReturn(0.50);

        sampler.sample(new double[] { -0.1 }, 101);
        sampler.sample(new double[] { 11.1 }, 102);
        sampler.sample(new double[] { 99.8 }, 103);

        double[] expectedScores = new double[3];
        expectedScores[0] = -lambda * 101L + Math.log(-Math.log(0.25));
        expectedScores[1] = -lambda * 102L + Math.log(-Math.log(0.75));
        expectedScores[2] = -lambda * 103L + Math.log(-Math.log(0.50));
        Arrays.sort(expectedScores);

        List<WeightedPoint> samples = sampler.getWeightedSamples();
        samples.sort(Comparator.comparing(WeightedPoint::getWeight));

        for (int i = 0; i < 3; i++) {
            assertEquals(expectedScores[i], samples.get(i).getWeight(), EPSILON);
        }
    }

    @Test
    public void testIsReadyIsFull() {
        WeightedPoint sampledPoint;
        int i;
        for (i = 1; i < sampleSize / 4; i++) {
            sampledPoint = sampler.sample(new double[] { Math.random() }, i);
            assertFalse(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertNotNull(sampledPoint);
            assertNull(sampler.getEvictedPoint());
        }

        for (i = sampleSize / 4; i < sampleSize; i++) {
            sampledPoint = sampler.sample(new double[] { Math.random() }, i);
            assertTrue(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertNotNull(sampledPoint);
            assertNull(sampler.getEvictedPoint());
        }

        sampledPoint = sampler.sample(new double[] { Math.random() }, sampleSize);
        assertTrue(sampler.isReady());
        assertTrue(sampler.isFull());
        assertEquals(i, sampler.getSize());
        assertNotNull(sampledPoint);
        assertNull(sampler.getEvictedPoint());

        WeightedPoint evictedPoint;

        for (i = sampleSize + 1; i < 2 * sampleSize; i++) {
            sampledPoint = sampler.sample(new double[] { Math.random() }, i);
            evictedPoint = sampler.getEvictedPoint();
            assertTrue(sampler.isReady());
            assertTrue(sampler.isFull());
            assertEquals(sampleSize, sampler.getSize());

            // Either the sampled point and the evicted point are both null or both non-null
            assertEquals(sampledPoint == null, evictedPoint == null);
        }
    }

    @Test
    public void testSample() {
        // first populate the sampler:

        int entriesSeen;
        for (entriesSeen = 0; entriesSeen < 2 * sampleSize; entriesSeen++) {
            sampler.sample(new double[] { Math.random() }, entriesSeen);
        }

        assertEquals(entriesSeen, 2 * sampleSize);
        assertTrue(sampler.isFull());

        // find the lowest weight currently in the sampler
        double maxWeight = sampler.getWeightedSamples().stream().mapToDouble(WeightedPoint::getWeight).max()
                .orElseThrow(IllegalStateException::new);

        // First choose a random value U so that
        // -lambda * entriesSeen + log(-log(U)) > maxWeight
        // which is equivalent to
        // U < exp(-exp(maxWeight + lambda * entriesSeen))
        // using this formula results in an underflow, so just use a very small number

        double u = 10e-100;
        when(random.nextDouble()).thenReturn(u);

        // With this choice of u, the next sample should be rejected
        WeightedPoint sampledPoint = sampler.sample(new double[] { Math.random() }, ++entriesSeen);
        assertNull(sampledPoint);
        assertNull(sampler.getEvictedPoint());

        double maxWeight2 = sampler.getWeightedSamples().stream().mapToDouble(WeightedPoint::getWeight).max()
                .orElseThrow(IllegalStateException::new);

        assertEquals(maxWeight, maxWeight2);

        // Next choose a large value of u (i.e., close to 1)
        // For this choice of U, the new point should be accepted

        u = 1 - 10e-100;
        when(random.nextDouble()).thenReturn(u);

        double point = Math.random();
        sampledPoint = sampler.sample(new double[] { point }, ++entriesSeen);

        assertNotNull(sampledPoint);
        assertArrayEquals(new double[] { point }, sampledPoint.getPoint());
        assertEquals(entriesSeen, sampledPoint.getSequenceIndex());
        assertTrue(sampledPoint.getWeight() < sampler.getEvictedPoint().getWeight());
        assertNotNull(sampler.getEvictedPoint());
        assertEquals(maxWeight, sampler.getEvictedPoint().getWeight(), EPSILON);

        maxWeight2 = sampler.getWeightedSamples().stream().mapToDouble(WeightedPoint::getWeight).max()
                .orElseThrow(IllegalStateException::new);
        assertTrue(maxWeight2 < maxWeight);
        assertTrue(sampler.getSamples().stream().anyMatch(array -> array[0] == point));
    }
}
