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

/**
 * This container class stores a point along with a weight value and a sequence
 * index. This class is used by SimpleStreamSampler, where the weight value
 * determines which points in the sample are evicted when new points are added.
 */
public final class WeightedPoint {

	/**
	 * The point values.
	 */
	private final double[] point;

	/**
	 * A weight value.
	 */
	private final double weight;

	/**
	 * An ordinal value corresponding to when this point was added to the forest.
	 */
	private final long sequenceIndex;

	/**
	 * Construct a new WeightedPoint.
	 *
	 * @param point
	 *            The point values.
	 * @param sequenceIndex
	 *            An ordinal value corresponding to when this point was added to the
	 *            forest.
	 * @param weight
	 *            A weight value.
	 */
	public WeightedPoint(double[] point, final long sequenceIndex, final double weight) {
		this.point = point;
		this.weight = weight;
		this.sequenceIndex = sequenceIndex;
	}

	/**
	 * @return the point values.
	 */
	public double[] getPoint() {
		return point;
	}

	/**
	 * @return the weight value.
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * @return the sequence index, an ordinal value corresponding to when this point
	 *         was added to the forest.
	 */
	public long getSequenceIndex() {
		return sequenceIndex;
	}
}
