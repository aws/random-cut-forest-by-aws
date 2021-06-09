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
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_INITIAL_ACCEPT_FRACTION;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;

import java.util.Random;

import com.amazon.randomcutforest.config.Config;

public abstract class AbstractStreamSampler<P> implements IStreamSampler<P> {
    /**
     * The decay factor used for generating the weight of the point. For greater
     * values of timeDecay we become more biased in favor of recent points.
     */
    protected double timeDecay;

    /**
     * The sequence index corresponding to the most recent change to
     * {@code timeDecay}.
     */
    protected long mostRecentTimeDecayUpdate = 0;

    /**
     * most recent timestamp, used to determine lastUpdateOfTimeDecay
     */
    protected long maxSequenceIndex = 0;

    /**
     * The accumulated sum of timeDecay before the last update
     */
    protected double accumuluatedTimeDecay = 0;

    /**
     * The random number generator used in sampling.
     */
    protected ReplayableRandom random;

    /**
     * The point evicted by the last call to {@link #update}, or null if the new
     * point was not accepted by the sampler.
     */
    protected transient ISampled<P> evictedPoint;

    /**
     * the fraction of points admitted to the sampler even when the sampler can
     * accept (not full) this helps control the initial behavior of the points and
     * ensure robustness by ensuring that the samplers do not all sample the initial
     * set of points.
     */
    protected final double initialAcceptFraction;

    /**
     * The number of points in the sample when full.
     */
    protected final int capacity;

    /**
     * This field is used to temporarily store the result from a call to
     * {@link #acceptPoint} for use in the subsequent call to {@link #addPoint}.
     *
     * Visible for testing.
     */
    protected AcceptPointState acceptPointState;

    public abstract boolean acceptPoint(long sequenceIndex);

    @Override
    public abstract void addPoint(P pointIndex);

    AbstractStreamSampler(Builder<?> builder) {
        this.capacity = builder.capacity;
        this.initialAcceptFraction = builder.initialAcceptFraction;
        this.timeDecay = builder.timeDecay;
        if (builder.random != null) {
            this.random = new ReplayableRandom(builder.random);
        } else {
            this.random = new ReplayableRandom(builder.randomSeed);
        }
    }

    /**
     * Weight is computed as <code>-log(w(i)) + log(-log(u(i))</code>, where
     *
     * <ul>
     * <li><code>w(i) = exp(timeDecay * sequenceIndex)</code></li>
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
        maxSequenceIndex = (maxSequenceIndex < sequenceIndex) ? sequenceIndex : maxSequenceIndex;
        return (float) (-(sequenceIndex - mostRecentTimeDecayUpdate) * timeDecay - accumuluatedTimeDecay
                + Math.log(-Math.log(randomNumber)));
    }

    /**
     * Sets the timeDecay on the fly. Note that the assumption is that the times
     * stamps corresponding to changes to timeDecay and sequenceIndexes are
     * non-decreasing -- the sequenceIndexes can be out of order among themselves
     * within two different times when timeDecay was changed.
     * 
     * @param newTimeDecay the new sampling rate
     */
    public void setTimeDecay(double newTimeDecay) {
        // accumulatedTimeDecay keeps track of adjustments and is zeroed out when the
        // arrays are exported for some reason
        accumuluatedTimeDecay += (maxSequenceIndex - mostRecentTimeDecayUpdate) * timeDecay;
        timeDecay = newTimeDecay;
        mostRecentTimeDecayUpdate = maxSequenceIndex;
    }

    /**
     * @return the time decay value that determines the rate of decay of previously
     *         seen points. Larger values of time decay indicate a greater bias
     *         toward recent points. A value of 0 corresponds to a uniform sample
     *         over the stream.
     */
    public double getTimeDecay() {
        return timeDecay;
    }

    public long getMaxSequenceIndex() {
        return maxSequenceIndex;
    }

    public void setMaxSequenceIndex(long index) {
        maxSequenceIndex = index;
    }

    public long getMostRecentTimeDecayUpdate() {
        return mostRecentTimeDecayUpdate;
    }

    public void setMostRecentTimeDecayUpdate(long index) {
        mostRecentTimeDecayUpdate = index;
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.TIME_DECAY.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    String.format("Setting '%s' must be a double value", name));
            setTimeDecay((Double) value);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public <T> T getConfig(String name, Class<T> clazz) {
        checkNotNull(clazz, "clazz must not be null");
        if (Config.TIME_DECAY.equals(name)) {
            checkArgument(clazz.isAssignableFrom(Double.class),
                    String.format("Setting '%s' must be a double value", name));
            return clazz.cast(getTimeDecay());
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    /**
     * @return the number of points contained by the sampler when full.
     */
    @Override
    public int getCapacity() {
        return capacity;
    }

    public double getInitialAcceptFraction() {
        return initialAcceptFraction;
    }

    public long getRandomSeed() {
        return random.randomSeed;
    }

    protected class ReplayableRandom {
        long randomSeed;
        Random testRandom;

        ReplayableRandom(long randomSeed) {
            this.randomSeed = randomSeed;
        }

        ReplayableRandom(Random random) {
            this.testRandom = random;
        }

        double nextDouble() {
            if (testRandom != null) {
                return testRandom.nextDouble();
            }
            Random newRandom = new Random(randomSeed);
            randomSeed = newRandom.nextLong();
            return newRandom.nextDouble();
        }
    }

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        protected int capacity = DEFAULT_SAMPLE_SIZE;
        protected double timeDecay = 0;
        protected Random random = null;
        protected long randomSeed = new Random().nextLong();
        protected long maxSequenceIndex = 0;
        protected long sequenceIndexOfMostRecentTimeDecayUpdate = 0;
        protected double initialAcceptFraction = DEFAULT_INITIAL_ACCEPT_FRACTION;

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T randomSeed(long seed) {
            this.randomSeed = seed;
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
            return (T) this;
        }

        public T maxSequenceIndex(long maxSequenceIndex) {
            this.maxSequenceIndex = maxSequenceIndex;
            return (T) this;
        }

        public T mostRecentTimeDecayUpdate(long sequenceIndexOfMostRecentTimeDecayUpdate) {
            this.sequenceIndexOfMostRecentTimeDecayUpdate = sequenceIndexOfMostRecentTimeDecayUpdate;
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
    }
}
