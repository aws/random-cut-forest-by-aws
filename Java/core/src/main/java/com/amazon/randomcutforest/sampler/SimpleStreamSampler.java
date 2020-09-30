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

import java.util.*;

import com.amazon.randomcutforest.executor.Sequential;

/**
 * SimpleStreamSamplerV2 is a sampler with a fixed sample size. Once the sampler
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
public class SimpleStreamSampler<P> implements IStreamSampler<P> {

    /**
     * A min-heap containing the weighted points currently in sample. The head
     * element is the lowest priority point in the sample (or, equivalently, is the
     * point with the greatest weight).
     */
    private final Queue<Weighted<P>> weightedSamples;

    /**
     * The number of points in the sample when full.
     */
    private final int sampleSize;
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
    private transient Sequential<P> evictedPoint;
    /**
     * A flag to determine if the sequence information is to be stored
     */
    private boolean storeSequenceIndices = false;

    public SimpleStreamSampler(final int sampleSize, final double lambda, Random random, boolean storeSequenceIndices) {
        this.sampleSize = sampleSize;
        entriesSeen = 0;
        weightedSamples = new PriorityQueue(Comparator.comparingDouble(Weighted<P>::getWeight).reversed());
        this.random = random;
        this.lambda = lambda;
        this.storeSequenceIndices = storeSequenceIndices;
    }

    public SimpleStreamSampler(int sampleSize, final double lambda, long seed, boolean storeSequenceIndices) {
        this(sampleSize, lambda, new Random(seed), storeSequenceIndices);
    }

    public SimpleStreamSampler(int sampleSize, final double lambda, long seed) {
        this(sampleSize, lambda, new Random(seed), false);
    }

    /**
     * Submit a new point to the sampler. When the point is submitted, a new weight
     * is computed for the point using the {@link #computeWeight} method. If the new
     * weight is smaller than the largest weight currently in the sampler, then the
     * new point is accepted into the sampler and the point corresponding to the
     * largest weight is evicted.
     *
     * @param seqIndex The timestamp value being submitted.
     * @return A weighted point that can be added to the sampler or null
     */
    public Optional<Double> acceptSample(long seqIndex) {
        evictedPoint = null;
        double weight = computeWeight(seqIndex);
        ++entriesSeen;

        if (entriesSeen <= sampleSize || weight < weightedSamples.element().getWeight()) {
            if (isFull()) {
                Weighted<P> tmp = weightedSamples.poll();
                if (storeSequenceIndices) {
                    checkState(tmp.getClass() == Sequential.class, "incorrect use");
                    evictedPoint = (Sequential<P>) tmp;
                } else {
                    evictedPoint = new Sequential(tmp.getValue(), tmp.getWeight(), 1L);
                }
            }
            return Optional.of(weight);
        }
        return Optional.empty();
    }

    /**
     * adds the sample to sampler; if the sampler was full, then the sampler has
     * already evicted a point in determining the weight.
     * 
     * @param point  to be entered in sampler
     * @param weight computed by acceptSample
     * @param seqNum timestamp
     */

    @Override
    public void addSample(P point, double weight, long seqNum) {
        if (storeSequenceIndices) {
            weightedSamples.add(new Sequential(point, weight, seqNum));
        } else {
            weightedSamples.add(new Weighted(point, weight));
        }
        checkState(weightedSamples.size() <= sampleSize,
                "The number of points in the sampler is greater than the sample size");
    }

    /**
     * The basic sampling broken down into a proposal/acceptance and a commit. This
     * breakdown allows the Tree to be in sync with the sampler when duplicates are
     * present.
     *
     * @param point  entry for sample
     * @param seqNum sequential stamp
     * @return true if the point is accepted and added, false otherwise
     */
    public boolean sample(P point, long seqNum) {
        Optional<Double> preSample = acceptSample(seqNum);
        if (preSample.isPresent()) {
            addSample(point, preSample.get(), seqNum);
            return true;
        }
        return false;
    }

    /**
     * @return the point evicted by the most recent call to {@link #sample}, or null
     *         if no point was evicted.
     */
    @Override
    public Optional<Sequential<P>> getEvictedPoint() {
        return Optional.ofNullable(evictedPoint);
    }

    /**
     * @return the list of weighted points currently in the sample. If there is no
     *         sequential information then a dummy variable is placed.
     */
    @Override
    public List<Weighted<P>> getWeightedSamples() {
        ArrayList<Weighted<P>> result;
        if (!storeSequenceIndices) {
            result = new ArrayList<>(weightedSamples);
        } else {
            result = new ArrayList<>();
            weightedSamples.stream().map(e -> result.add(new Weighted(e.getValue(), e.getWeight())));
        }
        return result;

    }

    /**
     * @return the list of weighted points currently in the sample. If there is no
     *         sequential information then a dummy variable is placed.
     */
    public List<Sequential<P>> getSequentialSamples() {
        ArrayList<Sequential<P>> result = new ArrayList<>();
        for (Weighted<P> e : weightedSamples) {
            if (storeSequenceIndices) {
                result.add((Sequential<P>) e);
            } else {
                result.add(new Sequential(e.getValue(), e.getWeight(), 1L));
            }
        }
        return result;
    }

    /**
     * @return true if this sampler contains enough points to support the anomaly
     *         score computation, false otherwise.
     */
    public boolean isReady() {
        return weightedSamples.size() >= sampleSize / 4;
    }

    /**
     * @return true if the sampler has reached it's full capacity, false otherwise.
     */
    public boolean isFull() {
        return weightedSamples.size() == sampleSize;
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
    protected double computeWeight(long sequenceIndex) {
        double randomNumber = 0d;
        while (randomNumber == 0d) {
            randomNumber = random.nextDouble();
        }

        return -sequenceIndex * lambda + Math.log(-Math.log(randomNumber));
    }

    /**
     * @return the number of points contained by the sampler when full.
     */
    public long getCapacity() {
        return sampleSize;
    }

    /**
     * @return the number of points currently contained by the sampler.
     */
    public long getSize() {
        return weightedSamples.size();
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
}
