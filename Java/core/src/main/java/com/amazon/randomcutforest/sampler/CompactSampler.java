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

import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import com.amazon.randomcutforest.executor.Sequential;

/**
 * SimpleStreamSampler is a sampler with a fixed sample size. Once the sampler
 * is full, when a new point is submitted to the sampler decision is made to
 * accept or reject the new point. If the point is accepted, then an older point
 * is removed from the sampler. This class implements time-based reservoir
 * sampling, which means that newer points are given more weight than older
 * points in the randomized decision.
 * <p>
 * The sampler algorithm is an example of the general weighted reservoir
 * sampling algorithm, which works like this:
 *
 * <ol>
 * <li>For each item i choose a random number u(i) uniformly from the interval
 * (0, 1) and compute the weight function <code>-(1 / c(i)) * log u(i)</code>,
 * for a given coefficient function c(i).</li>
 * <li>For a sample size of N, maintain a list of the N items with the smallest
 * weights.</li>
 * <li>When a new item is submitted to sampler, compute its weight. If it is
 * smaller than the largest weight currently contained in the sampler, then the
 * item with the largest weight is evicted from the sample and replaced by the
 * new item.</li>
 * </ol>
 * <p>
 * The SimpleStreamSampler creates a time-decayed sample by using the
 * coefficient function: <code>c(i) = exp(lambda * sequenceIndex(i))</code>.
 */
public class CompactSampler implements IStreamSampler<Integer> {

    /**
     * A min-heap containing the weighted points currently in sample. The head
     * element is the lowest priority point in the sample (or, equivalently, is the
     * point with the greatest weight).
     */
    protected final float[] weightArray;

    protected final int[] referenceArray;

    protected final long[] sequenceArray;
    /**
     * The number of points in the sample when full.
     */
    protected final int maxSize;

    protected int currentSize;
    /**
     * The decay factor used for generating the weight of the point. For greater
     * values of lambda we become more biased in favor of recent points.
     */
    private final double lambda;
    /**
     * The random number generator used in sampling.
     */
    private final Random random;
    /**
     * The number of points which have been submitted to the update method.
     */
    private long entriesSeen;
    /**
     * The point evicted by the last call to {@link #sample}, or if the new point
     * was not accepted by the sampler.
     */
    private transient Sequential<Integer> evictedPoint;

    private final boolean storeSequenceEnabled;

    /**
     * Construct a new CompactSampler.
     *
     * @param sampleSize The number of points in the sampler when full.
     * @param lambda     The decay factor used for generating the weight of the
     *                   point. For greater values of lambda we become more biased
     *                   in favor of recent points.
     * @param seed       The seed value used to create a random number generator.
     */
    public CompactSampler(final int sampleSize, final double lambda, long seed, boolean storeSeq) {
        this(sampleSize, lambda, new Random(seed), storeSeq);
    }

    /**
     * Construct a new OffsetSampler. This constructor exposes the Random argument
     * so that it can be mocked for testing.
     *
     * @param sampleSize The number of points in the sampler when full.
     * @param lambda     The decay factor used for generating the weight of the
     *                   point. For greater values of lambda we become more biased
     *                   in favor of recent points.
     * @param random     A random number generator that will be used in sampling.
     */
    protected CompactSampler(final int sampleSize, final double lambda, Random random, boolean storeSeq) {
        this.maxSize = sampleSize;
        entriesSeen = 0;
        currentSize = 0;
        weightArray = new float[sampleSize];
        referenceArray = new int[sampleSize];
        this.storeSequenceEnabled = storeSeq;
        if (storeSeq) {
            this.sequenceArray = new long[sampleSize];
        } else {
            this.sequenceArray = null;
        }
        this.random = random;
        this.lambda = lambda;
    }

    /**
     * This convenience constructor creates a SimpleStreamSampler with lambda equal
     * to 0, which is equivalent to uniform sampling on the stream.
     *
     * @param sampleSize The number of points in the sampler when full.
     * @param seed       The seed value used to create a random number generator.
     * @return a new SimpleStreamSampler which samples uniformly from its input.
     */
    public static CompactSampler uniformSampler(int sampleSize, long seed, boolean storeSeq) {
        return new CompactSampler(sampleSize, 0.0, seed, storeSeq);
    }

    @Override
    public Optional<Float> acceptSample(long sequenceIndex) {
        evictedPoint = null;
        float weight = computeWeight(sequenceIndex);
        entriesSeen++;
        if (currentSize < maxSize) {
            return Optional.of(weight);
        } else if (weight < weightArray[0]) {
            long tmpLong = 0;
            if (storeSequenceEnabled) {
                tmpLong = sequenceArray[0];
            }
            evictedPoint = new Sequential(referenceArray[0], weightArray[0], tmpLong);
            --currentSize;
            weightArray[0] = weightArray[currentSize];
            referenceArray[0] = referenceArray[currentSize];
            swapDown(0);
            return Optional.of(weight);
        } else {
            return Optional.empty();
        }

    }

    private void swapDown(int startIndex) {
        int current = startIndex;
        while ((2 * current + 1) < currentSize) {
            int maxIndex = 2 * current + 1;
            if ((2 * current + 2 < currentSize) && (weightArray[2 * current + 2] > weightArray[maxIndex]))
                maxIndex = 2 * current + 2;
            if (weightArray[maxIndex] > weightArray[current]) {
                swapWeights(current, maxIndex);
                current = maxIndex;
            } else
                break;
        }
    }

    private void reheap() {
        for (int i = (currentSize + 1) / 2; i >= 0; i--) {
            swapDown(i);
        }
    }

    @Override
    public void addSample(Integer pointRef, float weight, long seqNUm) {
        checkState(currentSize < maxSize, " sampler full");
        weightArray[currentSize] = weight;
        referenceArray[currentSize] = pointRef;
        if (storeSequenceEnabled) {
            sequenceArray[currentSize] = seqNUm;
        }
        int current = currentSize++;
        while (current > 0) {
            int tmp = (current - 1) / 2;
            if (weightArray[tmp] < weightArray[current]) {
                swapWeights(current, tmp);
                current = tmp;
            } else
                break;
        }
    }

    @Override
    public List<Sequential<Integer>> getSequentialSamples() {
        checkState(storeSequenceEnabled == true, "incorrect option");
        List<Sequential<Integer>> result = new ArrayList<>();
        for (int i = 0; i < currentSize; i++) {
            result.add(new Sequential(referenceArray[i], weightArray[i], sequenceArray[i]));
        }
        return result;
    }

    @Override
    public List<Weighted<Integer>> getWeightedSamples() {
        List<Weighted<Integer>> result = new ArrayList<>();
        for (int i = 0; i < currentSize; i++) {
            result.add(new Weighted<>(referenceArray[i], weightArray[i]));
        }
        return result;
    }

    /**
     * @return the point evicted by the most recent call to {@link #sample}, or null
     *         if no point was evicted.
     */
    public Optional<Sequential<Integer>> getEvictedPoint() {
        return Optional.ofNullable(evictedPoint);
    }

    /**
     * @return true if this sampler contains enough points to support the anomaly
     *         score computation, false otherwise.
     */
    public boolean isReady() {
        return entriesSeen >= maxSize / 4;
    }

    /**
     * @return true if the sampler has reached it's full capacity, false otherwise.
     */
    public boolean isFull() {
        return entriesSeen >= maxSize;
    }

    /**
     * Score is computed as <code>-log(w(i)) + log(-log(u(i))</code>, where
     *
     * <ul>
     * <li><code>w(i) = exp(lambda * sequenceIndex)</code></li>
     * <li><code>u(i)</code> is chosen uniformly from (0, 1)</li>
     * </ul>
     * <p>
     * A higher score means lower priority. So the points with the lower score have
     * higher chance of making it to the sample.
     *
     * @param sequenceIndex The sequenceIndex of the point whose score is being
     *                      computed.
     * @return the weight value used to define point priority
     */
    protected float computeWeight(long sequenceIndex) {
        double randomNumber = 0d;
        while (randomNumber == 0d) {
            randomNumber = random.nextDouble();
        }

        return (float) (-sequenceIndex * lambda + Math.log(-Math.log(randomNumber)));
    }

    /**
     * @return the number of points contained by the sampler when full.
     */
    public long getCapacity() {
        return maxSize;
    }

    /**
     * @return the number of points currently contained by the sampler.
     */
    public long getSize() {
        return currentSize;
    }

    /**
     * @return the lambda value that determines the amount of bias given toward
     *         recent points. Larger values of lambda indicate a greater bias toward
     *         recent points. A value of 0 corresponds to a uniform sample over the
     *         stream.
     */
    public double getLambda() {
        return lambda;
    }

    private void swapWeights(int a, int b) {
        int tmp = referenceArray[a];
        referenceArray[a] = referenceArray[b];
        referenceArray[b] = tmp;

        float tmpDouble = weightArray[a];
        weightArray[a] = weightArray[b];
        weightArray[b] = tmpDouble;

        if (storeSequenceEnabled) {
            long tmpLong = sequenceArray[a];
            sequenceArray[a] = sequenceArray[b];
            sequenceArray[b] = tmpLong;
        }
    }

    public void reInitialize(CompactSamplerData samplerData) {
        checkState(maxSize >= samplerData.maxSize, " need larger samplers");
        checkState(!storeSequenceEnabled || samplerData.sequenceArray != null, "sequences missing");
        currentSize = samplerData.currentSize;
        for (int i = 0; i < currentSize; i++) {
            referenceArray[i] = samplerData.referenceArray[i];
            weightArray[i] = samplerData.weightArray[i];
            if (storeSequenceEnabled) {
                sequenceArray[i] = samplerData.sequenceArray[i];
            }
        }
        reheap();
    }
}
