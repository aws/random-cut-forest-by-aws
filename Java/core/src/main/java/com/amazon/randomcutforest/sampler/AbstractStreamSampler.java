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

import java.util.Random;

import com.amazon.randomcutforest.config.Config;

public abstract class AbstractStreamSampler<P> implements IStreamSampler<P> {
    /**
     * The decay factor used for generating the weight of the point. For greater
     * values of lambda we become more biased in favor of recent points.
     */
    protected double timeDecay;

    /**
     * The sequence index corresponding to the most recent change to
     * {@code timeDecay}.
     */
    protected long mostRecentTimeDecayUpdate = 0;

    /**
     * most recent timestamp, used to determine lastUpdateOfLambda
     */
    protected long maxSequenceIndex = 0;

    /**
     * The accumulated sum of lambda before the last update
     */
    protected double accumuluatedTimeDecay = 0;

    /**
     * The random number generator used in sampling.
     */
    protected Random random;

    /**
     * The point evicted by the last call to {@link #update}, or null if the new
     * point was not accepted by the sampler.
     */
    protected transient ISampled<Integer> evictedPoint;

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

    /**
     * Weight is computed as <code>-log(w(i)) + log(-log(u(i))</code>, where
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
        maxSequenceIndex = (maxSequenceIndex < sequenceIndex) ? sequenceIndex : maxSequenceIndex;
        return (float) (-(sequenceIndex - mostRecentTimeDecayUpdate) * timeDecay - accumuluatedTimeDecay
                + Math.log(-Math.log(randomNumber)));
    }

    /**
     * sets the lambda on the fly. Note that the assumption is that the times stamps
     * corresponding to changes to lambda and sequenceIndexes are non-decreasing --
     * the sequenceIndexes can be out of order among themselves within two different
     * times when lambda was changed.
     * 
     * @param newLambda the new sampling rate
     */
    public void setTimeDecay(double newLambda) {
        // accumulatedLambda keeps track of adjustments and is zeroed out when the
        // arrays are
        // exported for some reason
        accumuluatedTimeDecay += (maxSequenceIndex - mostRecentTimeDecayUpdate) * timeDecay;
        timeDecay = newLambda;
        mostRecentTimeDecayUpdate = maxSequenceIndex;
    }

    /**
     * @return the lambda value that determines the rate of decay of previously seen
     *         points. Larger values of lambda indicate a greater bias toward recent
     *         points. A value of 0 corresponds to a uniform sample over the stream.
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
}
