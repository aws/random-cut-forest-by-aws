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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * <p>
 * CompactSampler is an implementation of time-based reservoir sampling. When a
 * point is submitted to the sampler, the decision to accept the point gives
 * more weight to newer points compared to older points. The newness of a point
 * is determined by its sequence index, and larger sequence indexes are
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
 * The coefficient function used by CompactSampler is:
 * <code>c(i) = exp(timeDecay * sequenceIndex(i))</code>.
 * </p>
 * <p>
 * Note that the sampling algorithm used is the same as for
 * {@link SimpleStreamSampler}. The difference is that CompactSampler is
 * specialized to use Integers as the point reference type and the
 * implementation uses less runtime memory.
 * </p>
 */
public class CompactSampler extends AbstractStreamSampler<Integer> {

    /**
     * When creating a {@code CompactSampler}, the user has the option to disable
     * storing sequence indexes. If storing sequence indexes is disabled, then this
     * value is used for the sequence index in {@link ISampled} instances returned
     * by {@link #getSample()}, {@link #getWeightedSample()}, and
     * {@link #getEvictedPoint()}.
     */
    public static final long SEQUENCE_INDEX_NA = -1L;

    /**
     * A max-heap containing the weighted points currently in sample. The head
     * element is the lowest priority point in the sample (or, equivalently, is the
     * point with the greatest weight).
     */
    protected final float[] weight;

    /**
     * Index values identifying the points in the sample. See
     * {@link com.amazon.randomcutforest.store.IPointStore}.
     */
    protected final int[] pointIndex;

    /**
     * Sequence indexes of the points in the sample.
     */
    protected final long[] sequenceIndex;

    /**
     * The number of points currently in the sample.
     */
    protected int size;

    /**
     * If true, then the sampler will store sequence indexes along with the sampled
     * points.
     */
    private final boolean storeSequenceIndexesEnabled;

    public static Builder<?> builder() {
        return new Builder<>();
    }

    public static CompactSampler uniformSampler(int sampleSize, long randomSeed, boolean storeSequences) {
        return new Builder<>().capacity(sampleSize).timeDecay(0).randomSeed(randomSeed)
                .storeSequenceIndexesEnabled(storeSequences).build();
    }

    protected CompactSampler(Builder<?> builder) {
        super(builder);
        checkArgument(builder.initialAcceptFraction > 0, " the admittance fraction cannot be <= 0");
        checkArgument(builder.capacity > 0, " sampler capacity cannot be <=0 ");

        this.storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        this.timeDecay = builder.timeDecay;
        this.maxSequenceIndex = builder.maxSequenceIndex;
        this.mostRecentTimeDecayUpdate = builder.sequenceIndexOfMostRecentTimeDecayUpdate;

        if (builder.weight != null || builder.pointIndex != null || builder.sequenceIndex != null
                || builder.validateHeap) {
            checkArgument(builder.weight != null && builder.weight.length == builder.capacity, " incorrect state");
            checkArgument(builder.pointIndex != null && builder.pointIndex.length == builder.capacity,
                    " incorrect state");
            checkArgument(
                    !builder.storeSequenceIndexesEnabled
                            || builder.sequenceIndex != null && builder.sequenceIndex.length == builder.capacity,
                    " incorrect state");
            this.weight = builder.weight;
            this.pointIndex = builder.pointIndex;
            this.sequenceIndex = builder.sequenceIndex;
            size = builder.size;
            reheap(builder.validateHeap);
        } else {
            checkArgument(builder.size == 0, "incorrect state");
            size = 0;
            weight = new float[builder.capacity];
            pointIndex = new int[builder.capacity];
            if (storeSequenceIndexesEnabled) {
                this.sequenceIndex = new long[builder.capacity];
            } else {
                this.sequenceIndex = null;
            }
        }
    }

    @Override
    public boolean acceptPoint(long sequenceIndex) {
        checkState(sequenceIndex >= mostRecentTimeDecayUpdate, "incorrect sequences submitted to sampler");
        evictedPoint = null;
        float weight = computeWeight(sequenceIndex);
        boolean initial = (size < capacity && random.nextDouble() < initialAcceptProbability(size));
        if (initial || (weight < this.weight[0])) {
            acceptPointState = new AcceptPointState(sequenceIndex, weight);
            if (!initial) {
                evictMax();
            }
            return true;
        }
        return false;
    }

    /**
     * evicts the maximum weight point from the sampler. can be used repeatedly to
     * change the size of the sampler and associated tree
     */
    public void evictMax() {
        long evictedIndex = storeSequenceIndexesEnabled ? this.sequenceIndex[0] : 0L;
        evictedPoint = new Weighted<>(this.pointIndex[0], this.weight[0], evictedIndex);
        --size;
        this.weight[0] = this.weight[size];
        this.pointIndex[0] = this.pointIndex[size];
        if (storeSequenceIndexesEnabled) {
            this.sequenceIndex[0] = this.sequenceIndex[size];
        }
        swapDown(0);
    }

    /**
     * Check to see if the weight at current index is greater than or equal to the
     * weight at each corresponding child index. If validate is true then throw an
     * IllegalStateException, otherwise swap the nodes and perform the same check at
     * the next level. Continue until you reach a level where the parent node's
     * weight is greater than or equal to both children's weights, or until there
     * are no more levels to descend.
     *
     * @param startIndex The index of node to start with.
     * @param validate   If true, a violation of the heap property will throw an
     *                   IllegalStateException. If false, then swap nodes that
     *                   violate the heap property.
     */
    private void swapDown(int startIndex, boolean validate) {
        int current = startIndex;
        while (2 * current + 1 < size) {
            int maxIndex = 2 * current + 1;
            if (2 * current + 2 < size && weight[2 * current + 2] > weight[maxIndex]) {
                maxIndex = 2 * current + 2;
            }
            if (weight[maxIndex] > weight[current]) {
                if (validate) {
                    throw new IllegalStateException("the heap property is not satisfied at index " + current);
                }
                swapWeights(current, maxIndex);
                current = maxIndex;
            } else {
                break;
            }
        }
    }

    private void swapDown(int startIndex) {
        swapDown(startIndex, false);
    }

    public void reheap(boolean validate) {
        for (int i = (size + 1) / 2; i >= 0; i--) {
            swapDown(i, validate);
        }
    }

    @Override
    public void addPoint(Integer pointIndex) {
        if (pointIndex != null) {
            checkState(size < capacity, "sampler full");
            checkState(acceptPointState != null,
                    "this method should only be called after a successful call to acceptSample(long)");
            this.weight[size] = acceptPointState.getWeight();
            this.pointIndex[size] = pointIndex;
            if (storeSequenceIndexesEnabled) {
                this.sequenceIndex[size] = acceptPointState.getSequenceIndex();
            }
            int current = size++;
            while (current > 0) {
                int tmp = (current - 1) / 2;
                if (this.weight[tmp] < this.weight[current]) {
                    swapWeights(current, tmp);
                    current = tmp;
                } else
                    break;
            }
            acceptPointState = null;
        }
    }

    /**
     * Return the list of sampled points. If this sampler was created with the
     * {@code storeSequenceIndexesEnabled} flag set to false, then all sequence
     * indexes in the list will be set to {@link #SEQUENCE_INDEX_NA}.
     *
     * @return the list of sampled points.
     */
    @Override
    public List<ISampled<Integer>> getSample() {
        return streamSample().collect(Collectors.toList());
    }

    /**
     * Return the list of sampled points with weights.
     * 
     * @return the list of sampled points with weights.
     */
    public List<Weighted<Integer>> getWeightedSample() {
        return streamSample().collect(Collectors.toList());
    }

    private Stream<Weighted<Integer>> streamSample() {
        reset_weights();
        return IntStream.range(0, size).mapToObj(i -> {
            long index = sequenceIndex != null ? sequenceIndex[i] : SEQUENCE_INDEX_NA;
            return new Weighted<>(pointIndex[i], weight[i], index);
        });
    }

    /**
     * removes the adjustments to weight in accumulated timeDecay and resets the
     * updates to timeDecay
     */
    private void reset_weights() {
        if (accumuluatedTimeDecay == 0)
            return;
        // now the weight computation of every element would not see this subtraction
        // which implies that every existing element should see the offset as addition
        for (int i = 0; i < size; i++) {
            weight[i] += accumuluatedTimeDecay;
        }
        accumuluatedTimeDecay = 0;
    }

    /**
     * @return the point evicted by the most recent call to {@link #update}, or null
     *         if no point was evicted.
     */
    public Optional<ISampled<Integer>> getEvictedPoint() {
        return Optional.ofNullable(evictedPoint);
    }

    /**
     * @return the number of points currently contained by the sampler.
     */
    @Override
    public int size() {
        return size;
    }

    public float[] getWeightArray() {
        return weight;
    }

    public int[] getPointIndexArray() {
        return pointIndex;
    }

    public long[] getSequenceIndexArray() {
        return sequenceIndex;
    }

    public boolean isStoreSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    private void swapWeights(int a, int b) {
        int tmp = pointIndex[a];
        pointIndex[a] = pointIndex[b];
        pointIndex[b] = tmp;

        float tmpDouble = weight[a];
        weight[a] = weight[b];
        weight[b] = tmpDouble;

        if (storeSequenceIndexesEnabled) {
            long tmpLong = sequenceIndex[a];
            sequenceIndex[a] = sequenceIndex[b];
            sequenceIndex[b] = tmpLong;
        }
    }

    public static class Builder<T extends Builder<T>> extends AbstractStreamSampler.Builder<T> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        private int size = 0;
        private float[] weight = null;
        private int[] pointIndex = null;
        private long[] sequenceIndex = null;
        private boolean validateHeap = false;
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;

        public T size(int size) {
            this.size = size;
            return (T) this;
        }

        public T weight(float[] weight) {
            this.weight = weight;
            return (T) this;
        }

        public T pointIndex(int[] pointIndex) {
            this.pointIndex = pointIndex;
            return (T) this;
        }

        public T sequenceIndex(long[] sequenceIndex) {
            this.sequenceIndex = sequenceIndex;
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T validateHeap(boolean validateHeap) {
            this.validateHeap = validateHeap;
            return (T) this;
        }

        public CompactSampler build() {
            return new CompactSampler(this);
        }
    }
}
