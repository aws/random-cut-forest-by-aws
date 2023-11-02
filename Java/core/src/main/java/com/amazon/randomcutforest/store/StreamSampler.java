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

package com.amazon.randomcutforest.store;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INITIAL_ACCEPT_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Optional;
import java.util.Random;

import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.sampler.ISampled;
import com.amazon.randomcutforest.util.Weighted;

/**
 * The following class is a sampler for generic objects that allow weighted time
 * dependent sampling. It is an encapsulation of CompactSampler in
 * RandomCutForest core and is meant to be extended in multiple ways. Hence the
 * functions are protected and should be overriden/not used arbirarily.
 */
public class StreamSampler<P> {

    // basic time dependent sampler
    protected final CompactSampler sampler;

    // list of objects
    protected final ArrayList<Weighted<P>> objectList;

    // managing indices
    protected final IndexIntervalManager intervalManager;

    // accounting for evicted items
    protected Optional<P> evicted;

    // sequence number used in sequential sampling
    protected long sequenceNumber = -1L;

    // number of items seen, different from sequenceNumber in case of merge
    protected long entriesSeen = 0L;

    protected boolean currentlySampling;

    public static Builder<?> builder() {
        return new Builder<>();
    }

    public StreamSampler(Builder<?> builder) {
        sampler = new CompactSampler.Builder<>().capacity(builder.capacity)
                .storeSequenceIndexesEnabled(builder.storeSequenceIndexesEnabled).randomSeed(builder.randomSeed)
                .initialAcceptFraction(builder.initialAcceptFraction).timeDecay(builder.timeDecay).build();
        objectList = new ArrayList<>(builder.capacity);
        intervalManager = new IndexIntervalManager(builder.capacity);
        evicted = Optional.empty();
        currentlySampling = true;
    }

    /**
     * a basic sampling operation that accounts for weights of items. This function
     * will be overriden in future classes.
     * 
     * @param object to be sampled
     * @param weight weight of object (non-negative); although 0 weight implies do
     *               not sample
     * @return true if the object is sampled and false if the object is not sampled;
     *         if true then there may have been an eviction which is updated
     */
    protected boolean sample(P object, float weight) {
        ++sequenceNumber;
        ++entriesSeen;
        if (currentlySampling) {
            if (sampler.acceptPoint(sequenceNumber, weight)) {
                Optional<ISampled<Integer>> samplerEvicted = sampler.getEvictedPoint();

                if (samplerEvicted.isPresent()) {
                    int oldIndex = samplerEvicted.get().getValue();
                    evicted = Optional.of(objectList.get(oldIndex).index);
                    intervalManager.releaseIndex(oldIndex);
                }
                int index = intervalManager.takeIndex();
                if (index < objectList.size()) {
                    objectList.set(index, new Weighted<>(object, weight));
                } else {
                    objectList.add(new Weighted<>(object, weight));
                }
                sampler.addPoint(index);
                return true;
            }
        }
        evicted = Optional.empty();
        return false;
    }

    public StreamSampler(StreamSampler<P> first, StreamSampler<P> second, int capacity, double timeDecay, long seed) {
        checkArgument(capacity > 0, "capacity has to be positive");
        double initialAcceptFraction = max(first.sampler.getInitialAcceptFraction(),
                second.sampler.getInitialAcceptFraction());
        // merge would remove sequenceIndex information

        objectList = new ArrayList<>(capacity);
        int[] pointList = new int[capacity];
        float[] weightList = new float[capacity];
        intervalManager = new IndexIntervalManager(capacity);
        evicted = Optional.empty();
        currentlySampling = true;
        double firstUpdate = -(first.sampler.getMaxSequenceIndex() - first.sampler.getMostRecentTimeDecayUpdate())
                * first.sampler.getTimeDecay();
        ArrayList<Weighted<Integer>> list = new ArrayList<>();
        int offset = first.sampler.size();
        int[] firstList = first.sampler.getPointIndexArray();
        float[] firstWeightList = first.sampler.getWeightArray();
        for (int i = 0; i < offset; i++) {
            list.add(new Weighted<>(firstList[i], (float) (firstWeightList[i] + firstUpdate)));
        }
        double secondUpdate = -(second.sampler.getMaxSequenceIndex() - second.sampler.getMostRecentTimeDecayUpdate())
                * second.sampler.getTimeDecay();
        int secondOffset = second.sampler.size();
        int[] secondList = second.sampler.getPointIndexArray();
        float[] secondWeightList = second.sampler.getWeightArray();
        for (int i = 0; i < secondOffset; i++) {
            list.add(new Weighted<>(secondList[i] + offset, (float) (secondWeightList[i] + secondUpdate)));
        }
        list.sort((o1, o2) -> Float.compare(o1.weight, o2.weight));
        int size = min(capacity, list.size());
        for (int j = size - 1; j >= 0; j--) {
            int index = intervalManager.takeIndex();
            pointList[index] = index;
            weightList[index] = list.get(j).weight;
            if (list.get(j).index < offset) {
                objectList.add(first.objectList.get(list.get(j).index));
            } else {
                objectList.add(second.objectList.get(list.get(j).index - offset));
            }
        }
        // sequence number corresponds to linear order of time
        this.sequenceNumber = max(first.sequenceNumber, second.sequenceNumber);
        // entries seen is the sum
        this.entriesSeen = first.entriesSeen + second.entriesSeen;
        sampler = new CompactSampler.Builder<>().capacity(capacity).storeSequenceIndexesEnabled(false).randomSeed(seed)
                .initialAcceptFraction(initialAcceptFraction).timeDecay(timeDecay).pointIndex(pointList)
                .weight(weightList).randomSeed(seed).maxSequenceIndex(this.sequenceNumber)
                .mostRecentTimeDecayUpdate(this.sequenceNumber).build();
    }

    public boolean isCurrentlySampling() {
        return currentlySampling;
    }

    public void pauseSampling() {
        currentlySampling = false;
    }

    public void resumeSampling() {
        currentlySampling = true;
    }

    public ArrayList<Weighted<P>> getObjectList() {
        return objectList;
    }

    public int getCapacity() {
        return sampler.getCapacity();
    }

    public long getSequenceNumber() {
        return sequenceNumber;
    }

    public long getEntriesSeen() {
        return entriesSeen;
    }

    public static class Builder<T extends Builder<T>> {

        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        protected int capacity = DEFAULT_SAMPLE_SIZE;
        protected double timeDecay = 1.0 / (DEFAULT_SAMPLE_SIZE_COEFFICIENT_IN_TIME_DECAY * capacity);
        protected long randomSeed = new Random().nextLong();
        protected double initialAcceptFraction = DEFAULT_INITIAL_ACCEPT_FRACTION;

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T randomSeed(long seed) {
            this.randomSeed = seed;
            return (T) this;
        }

        public T initialAcceptFraction(double initialAcceptFraction) {
            this.initialAcceptFraction = initialAcceptFraction;
            return (T) this;
        }

        public T timeDecay(double timeDecay) {
            this.timeDecay = timeDecay;
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public StreamSampler build() {
            return new StreamSampler<>(this);
        }

        public double getTimeDecay() {
            return timeDecay;
        }

        public int getCapacity() {
            return capacity;
        }

        public long getRandomSeed() {
            return randomSeed;
        }
    }
}
