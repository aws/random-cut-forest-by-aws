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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.stream.Collectors;

import static com.amazon.randomcutforest.CommonUtils.checkState;

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
 * (0, 1) and compute the weight function <tt>-(1 / c(i)) * log u(i)</tt>, for a
 * given coefficient function c(i).</li>
 * <li>For a sample size of N, maintain a list of the N items with the smallest
 * weights.</li>
 * <li>When a new item is submitted to sampler, compute its weight. If it is
 * smaller than the largest weight currently contained in the sampler, then the
 * item with the largest weight is evicted from the sample and replaced by the
 * new item.</li>
 * </ol>
 * <p>
 * The SimpleStreamSampler creates a time-decayed sample by using the
 * coefficient function: <tt>c(i) = exp(lambda * sequenceIndex(i))</tt>.
 */
public class SimpleStreamSampler {

	/**
	 * Wraps PriorityQueue with default comparator.
	 *
	 * @param <WeightedPoint>
	 *            type of comparator.
	 */
	static class PriorityQueueWrapper<WeightedPoint> extends PriorityQueue<WeightedPoint> {

		/**
		 * Constructor of PriorityQueue with default WeightedPoint comparator.
		 */
		PriorityQueueWrapper() {
			super((Comparator<? super WeightedPoint>) POINT_COMPARATOR);
		}
	}

	/**
	 * This is the comparator used to order the priority queue in which we store the
	 * in-sample points. We want the head of the queue to be the element with the
	 * least priority, so we reverse the natural defined by the weight.
	 */
	static Comparator<WeightedPoint> POINT_COMPARATOR = Comparator.comparingDouble(WeightedPoint::getWeight).reversed();

	/**
	 * A min-heap containing the weighted points currently in sample. The head
	 * element is the lowest priority point in the sample (or, equivalently, is the
	 * point with the greatest weight).
	 */
	private final Queue<WeightedPoint> weightedSamples;

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
	private transient WeightedPoint evictedPoint;

	/**
	 * Construct a new SimpleStreamSampler.
	 *
	 * @param sampleSize
	 *            The number of points in the sampler when full.
	 * @param lambda
	 *            The decay factor used for generating the weight of the point. For
	 *            greater values of lambda we become more biased in favor of recent
	 *            points.
	 * @param seed
	 *            The seed value used to create a random number generator.
	 */
	public SimpleStreamSampler(final int sampleSize, final double lambda, long seed) {
		this(sampleSize, lambda, new Random(seed));
	}

	/**
	 * Construct a new SimpleStreamSampler. This constructor exposes the Random
	 * argument so that it can be mocked for testing.
	 *
	 * @param sampleSize
	 *            The number of points in the sampler when full.
	 * @param lambda
	 *            The decay factor used for generating the weight of the point. For
	 *            greater values of lambda we become more biased in favor of recent
	 *            points.
	 * @param random
	 *            A random number generator that will be used in sampling.
	 */
	protected SimpleStreamSampler(final int sampleSize, final double lambda, Random random) {
		this.sampleSize = sampleSize;
		entriesSeen = 0;
		weightedSamples = new PriorityQueueWrapper<>();
		this.random = random;
		this.lambda = lambda;
	}

	/**
	 * This convenience constructor creates a SimpleStreamSampler with lambda equal
	 * to 0, which is equivalent to uniform sampling on the stream.
	 *
	 * @param sampleSize
	 *            The number of points in the sampler when full.
	 * @param seed
	 *            The seed value used to create a random number generator.
	 * @return a new SimpleStreamSampler which samples uniformly from its input.
	 */
	public static SimpleStreamSampler uniformSampler(int sampleSize, long seed) {
		return new SimpleStreamSampler(sampleSize, 0.0, seed);
	}

	/**
	 * Submit a new point to the sampler. When the point is submitted, a new weight
	 * is computed for the point using the {@link #computeWeight} method. If the new
	 * weight is smaller than the largest weight currently in the sampler, then the
	 * new point is accepted into the sampler and the point corresponding to the
	 * largest weight is evicted.
	 *
	 * @param newPoint
	 *            A candidate point to add to the sampler.
	 * @param sequenceIndex
	 *            An ordinal index corresponding to when this point was added to the
	 *            forest.
	 * @return a WeightedPoint created from the input point if the input point is
	 *         accepted by the sampler. Return null otherwise.
	 */
	public WeightedPoint sample(double[] newPoint, long sequenceIndex) {
		evictedPoint = null;
		WeightedPoint candidate = null;
		double weight = computeWeight(sequenceIndex);
		++entriesSeen;

		if (entriesSeen <= sampleSize || weight < weightedSamples.element().getWeight()) {
			if (isFull()) {
				evictedPoint = weightedSamples.poll();
			}
			candidate = new WeightedPoint(newPoint, sequenceIndex, weight);
			weightedSamples.add(candidate);

			checkState(weightedSamples.size() <= sampleSize,
					"The number of points in the sampler is greater than the sample size");
		}

		return candidate;
	}

	/**
	 * @return the point evicted by the most recent call to {@link #sample}, or null
	 *         if no point was evicted.
	 */
	public WeightedPoint getEvictedPoint() {
		return evictedPoint;
	}

	/**
	 * @return the list of points currently in the sample.
	 */
	public List<double[]> getSamples() {
		return weightedSamples.stream().map(WeightedPoint::getPoint).collect(Collectors.toList());
	}

	/**
	 * @return the list of points currently in the sample.
	 */
	public List<WeightedPoint> getWeightedSamples() {
		return new ArrayList<>(weightedSamples);
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
	 * Score is computed as <tt>-log(w(i)) + log(-log(u(i))</tt>, where
	 *
	 * <ul>
	 * <li><tt>w(i) = exp(lambda * sequenceIndex)</tt></li>
	 * <li><tt>u(i)</tt> is chosen uniformly from (0, 1)</li>
	 * </ul>
	 * <p>
	 * A higher score means lower priority. So the points with the lower score have
	 * higher chance of making it to the sample.
	 *
	 * @param sequenceIndex
	 *            The sequenceIndex of the point whose score is being computed.
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
