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
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;

/**
 * <p>
 * SimpleStreamSampler is an implementation of time-based reservoir sampling.
 * When a point is submitted to the sampler, the decision to accept the point
 * gives more weight to newer points compared to older points. The newness of a
 * point is determined by its sequence index, and larger sequence indexes are
 * considered newer.
 * </p>
 * <p>
 * The sampler algorithm is an example of the general weighted reservoir
 * sampling algorithm, which works like this:
 * </p>
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
 * The coefficient function used by SimpleStreamSampler is:
 * <code>c(i) = exp(lambda * sequenceIndex(i))</code>.
 * </p>
 * <p>
 * For a specialized version of this algorithm that uses less runtime memory,
 * see {@link CompactSampler}.
 * </p>
 */
public class SimpleStreamSampler<P> extends AbstractStreamSampler<P> {

    /**
     * A min-heap containing the weighted points currently in sample. The head
     * element is the lowest priority point in the sample (or, equivalently, is the
     * point with the greatest weight).
     */
    private final Queue<Weighted<P>> sample;

    public SimpleStreamSampler(Builder<?> builder) {
        super(builder);
        sample = new PriorityQueue<>(Comparator.comparingDouble(Weighted<P>::getWeight).reversed());
        this.timeDecay = builder.timeDecay;
    }

    public SimpleStreamSampler(int sampleSize, double lambda, long seed, boolean storeSequenceIndexesEnabled) {
        this(new Builder<>().capacity(sampleSize).timeDecay(lambda).randomSeed(seed));
    }

    public SimpleStreamSampler(int sampleSize, double lambda, Random random, boolean storeSequenceIndexesEnabled) {
        this(new Builder<>().capacity(sampleSize).timeDecay(lambda).random(random));
    }

    public SimpleStreamSampler(int sampleSize, double lambda, long seed) {
        this(new Builder<>().capacity(sampleSize).timeDecay(lambda).randomSeed(seed));
    }

    /**
     * Submit a new point to the sampler. When the point is submitted, a new weight
     * is computed for the point using the computeWeight method. If the new weight
     * is smaller than the largest weight currently in the sampler, then the new
     * point is accepted into the sampler and the point corresponding to the largest
     * weight is evicted.
     *
     * @param sequenceIndex The timestamp value being submitted.
     * @return A weighted point that can be added to the sampler or null
     */
    public boolean acceptPoint(long sequenceIndex) {
        checkState(sequenceIndex >= mostRecentTimeDecayUpdate, "incorrect sequences submitted to sampler");

        evictedPoint = null;
        float weight = computeWeight(sequenceIndex);

        if ((sample.size() < capacity && random.nextDouble() < initialAcceptFraction + 1 - 1.0 * size() / capacity)
                || weight < sample.element().getWeight()) {
            if (isFull()) {
                evictedPoint = sample.poll();
            }
            acceptPointState = new AcceptPointState(sequenceIndex, weight);
            return true;
        }
        return false;
    }

    /**
     * adds the sample to sampler; if the sampler was full, then the sampler has
     * already evicted a point in determining the weight.
     * 
     * @param point to be entered in sampler
     */

    @Override
    public void addPoint(P point) {
        checkState(acceptPointState != null,
                "this method should only be called after a successful call to acceptSample(long)");
        sample.add(new Weighted<>(point, acceptPointState.getWeight(), acceptPointState.getSequenceIndex()));
        acceptPointState = null;
        checkState(sample.size() <= capacity, "The number of points in the sampler is greater than the sample size");
    }

    /**
     * Add a Weighted value directly to the sample. This method is intended to be
     * used to restore a sampler to a pre-existing state.
     * 
     * @param point A weighted point.
     */
    public void addSample(Weighted<P> point) {
        sample.add(point);
    }

    /**
     * @return the point evicted by the most recent call to {@link #update}, or null
     *         if no point was evicted.
     */
    @Override
    public Optional<ISampled<P>> getEvictedPoint() {
        return Optional.ofNullable(evictedPoint);
    }

    /**
     * @return the list of sampled points with weights.
     */
    public List<Weighted<P>> getWeightedSample() {
        return new ArrayList<>(sample);
    }

    /**
     * @return the list of sampled points.
     */
    @Override
    public List<ISampled<P>> getSample() {
        return new ArrayList<>(sample);
    }

    /**
     * @return the number of points currently contained by the sampler.
     */
    @Override
    public int size() {
        return sample.size();
    }

}
