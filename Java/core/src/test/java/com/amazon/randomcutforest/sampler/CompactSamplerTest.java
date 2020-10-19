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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.executor.Sequential;

public class CompactSamplerTest {

    private int sampleSize;
    private double lambda;
    private long seed;
    private Random random;
    private CompactSampler sampler;

    @BeforeEach
    public void setUp() {
        sampleSize = 101;
        lambda = 0.01;
        seed = 42L;
        random = spy(new Random(seed));
        sampler = new CompactSampler(sampleSize, lambda, random, false);
    }

    @Test
    public void testNew() {
        assertFalse(sampler.getEvictedPoint().isPresent());
        assertFalse(sampler.isReady());
        assertFalse(sampler.isFull());
        assertEquals(sampleSize, sampler.getCapacity());
        assertEquals(0, sampler.getSize());
        assertEquals(lambda, sampler.getLambda());

        CompactSampler uniformSampler = new CompactSampler(11, 0, 14, false);
        assertFalse(uniformSampler.getEvictedPoint().isPresent());
        assertFalse(uniformSampler.isReady());
        assertFalse(uniformSampler.isFull());
        assertEquals(11, uniformSampler.getCapacity());
        assertEquals(0, uniformSampler.getSize());
        assertEquals(0.0, uniformSampler.getLambda());
    }

    @Test
    public void testPointComparator() {
        Sequential<double[]> point1 = new Sequential(new double[] { 0.99, -55.2 }, 1.23, 999L);
        Sequential<double[]> point2 = new Sequential(new double[] { 2.2, 87.0 }, -77, 1000L);
        Sequential<double[]> point3 = new Sequential(new double[] { -2.1, 99.4 }, -77, 1001L);

        assertTrue(Comparator.comparingDouble(Weighted<double[]>::getWeight).reversed().compare(point1, point2) < 0);
        assertTrue(Comparator.comparingDouble(Weighted<double[]>::getWeight).reversed().compare(point3, point1) > 0);
        assertTrue(Comparator.comparingDouble(Weighted<double[]>::getWeight).reversed().compare(point2, point3) == 0);
    }

    @Test
    public void testGetScore() {
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
        // samples.sort(Comparator.comparing(Weighted::getWeight).reversed());

        // for (int i = 0; i < 3; i++) {
        // assertEquals(expectedScores[i], samples.get(i).getWeight(), EPSILON);
        // }
    }

    @Test
    public void testIsReadyIsFull() {
        int i;
        for (i = 1; i < sampleSize / 4; i++) {
            assertTrue(sampler.sample((int) Math.ceil(Math.random() * 100), i));
            assertFalse(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        for (i = sampleSize / 4; i < sampleSize; i++) {
            assertTrue(sampler.sample((int) Math.ceil(Math.random() * 100), i));
            assertTrue(sampler.isReady());
            assertFalse(sampler.isFull());
            assertEquals(i, sampler.getSize());
            assertFalse(sampler.getEvictedPoint().isPresent());
        }

        assertTrue(sampler.sample((int) Math.ceil(Math.random() * 100), sampleSize));
        assertTrue(sampler.isReady());
        assertTrue(sampler.isFull());
        assertEquals(i, sampler.getSize());
        assertFalse(sampler.getEvictedPoint().isPresent());

        java.util.Optional<Sequential<double[]>> evicted;
        for (i = sampleSize + 1; i < 2 * sampleSize; i++) {
            // Either the sampling and the evicted point are both null or both non-null
            assertTrue(
                    sampler.sample((int) Math.ceil(Math.random() * 100), i) == sampler.getEvictedPoint().isPresent());

            assertTrue(sampler.isReady());
            assertTrue(sampler.isFull());
            assertEquals(sampleSize, sampler.getSize());

        }
    }

    @Test
    public void testCompareSampling() {
        CompactSampler compact = new CompactSampler(10, 0.001, 10, false);
        SimpleStreamSampler<double[]> simple = new SimpleStreamSampler<>(10, 0.001, 10, false);

        for (int i = 0; i < 100000; i++) {
            Optional<Float> weightOne = compact.acceptSample(i);
            Optional<Float> weightTwo = simple.acceptSample(i);
            assertEquals(weightOne.isPresent(), weightTwo.isPresent());
            if (weightOne.isPresent()) {
                assertEquals(weightOne.get(), weightTwo.get(), 1E-10);
                compact.addSample(i, weightOne.get(), i);
                simple.addSample(new double[] { i }, weightTwo.get(), i);
            }
            assertEquals(compact.getEvictedPoint().isPresent(), simple.getEvictedPoint().isPresent());
            if (compact.getEvictedPoint().isPresent()) {
                // assertEquals(compact.getEvictedPoint().get().getWeight(),simple.getEvictedPoint().get().getWeight(),1E-10);
                int y = compact.getEvictedPoint().get().getValue();
                double[] x = simple.getEvictedPoint().get().getValue();
                assertEquals(x[0], y);
                assertEquals(compact.getEvictedPoint().get().getWeight(), simple.getEvictedPoint().get().getWeight(),
                        1E-10);

            }

        }
    }
    /*
     * @Test public void testSample() { // first populate the sampler:
     * 
     * int entriesSeen; for (entriesSeen = 0; entriesSeen < 2 * sampleSize;
     * entriesSeen++) { sampler.sample(new double[] { Math.random() }, entriesSeen);
     * }
     * 
     * assertEquals(entriesSeen, 2 * sampleSize); assertTrue(sampler.isFull());
     * 
     * // find the lowest weight currently in the sampler // double maxWeight = //
     * sampler.getWeightedSamples().stream().mapToDouble(Sequential::getWeight).max(
     * ) // .orElseThrow(IllegalStateException::new);
     * 
     * // First choose a random value U so that // -lambda * entriesSeen +
     * log(-log(U)) > maxWeight // which is equivalent to // U < exp(-exp(maxWeight
     * + lambda * entriesSeen)) // using this formula results in an underflow, so
     * just use a very small number
     * 
     * double u = 10e-100; when(random.nextDouble()).thenReturn(u);
     * 
     * // With this choice of u, the next sample should be rejected
     * 
     * assertFalse(sampler.sample(new double[] { Math.random() }, ++entriesSeen));
     * assertFalse(sampler.getEvictedPoint().isPresent());
     * 
     * // double maxWeight2 = //
     * sampler.getWeightedSamples().stream().mapToDouble(Sequential::getWeight).max(
     * ) // .orElseThrow(IllegalStateException::new);
     * 
     * // assertEquals(maxWeight, maxWeight2);
     * 
     * // Next choose a large value of u (i.e., close to 1) // For this choice of U,
     * the new point should be accepted
     * 
     * u = 1 - 10e-100; when(random.nextDouble()).thenReturn(u);
     * 
     * double point = Math.random(); Optional<Double> weight =
     * sampler.acceptSample(++entriesSeen); assertTrue(sampler.sample(new double[] {
     * point }, ++entriesSeen)); Optional<Sequential<double[]>> evicted =
     * sampler.getEvictedPoint(); assertTrue(evicted.isPresent());
     * assertTrue(weight.get() < evicted.get().getWeight()); //
     * assertEquals(maxWeight, evicted.get().getWeight(), EPSILON);
     * 
     * // maxWeight2 = //
     * sampler.getWeightedSamples().stream().mapToDouble(Sequential::getWeight).max(
     * ) // .orElseThrow(IllegalStateException::new); // assertTrue(maxWeight2 <
     * maxWeight); //
     * assertTrue(sampler.getWeightedSamples().stream().anyMatch((Sequential) array
     * // -> array.getValue() == point)); }
     */

}
