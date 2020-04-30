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

package com.amazon.randomcutforest.returntypes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collector;

/**
 * A Neighbor represents a point together with a distance, where the distance is
 * with respect to some query point. That is, we think of this point as being a
 * neighbor of the query point. If the feature is enabled in the forest, a
 * Neighbor will also contain a set of sequence indexes containing the times
 * this point was added to the forest.
 */
public class Neighbor {

	/**
	 * The neighbor point.
	 */
	public final double[] point;

	/**
	 * The distance between the neighbor point and the query point it was created
	 * from.
	 */
	public final double distance;

	/**
	 * A list of sequence indexes corresponding to the times when this neighbor
	 * point was added to the forest. If sequence indexes are not enabled for the
	 * forest, then this list will be empty.
	 */
	public final List<Long> sequenceIndexes;

	/**
	 * Create a new Neighbor.
	 *
	 * @param point
	 *            The neighbor point.
	 * @param distance
	 *            The distance between the neighbor point and the query point is was
	 *            created from.
	 * @param sequenceIndexes
	 *            A list of sequence indexes corresponding to the times when this
	 *            neighbor point was added to the forest.
	 */
	public Neighbor(double[] point, double distance, List<Long> sequenceIndexes) {
		this.point = point;
		this.distance = distance;
		this.sequenceIndexes = sequenceIndexes;
	}

	/**
	 * Get Neighbor collector which merges duplicate Neighbors and sorts them in
	 * ascending order of distance
	 *
	 * @return Neighbor collector
	 */
	public static Collector<Optional<Neighbor>, Map<Integer, Neighbor>, List<Neighbor>> collector() {
		return new CollectorImpl();
	}

	/**
	 * Merge sequence indexes of other Neighbor to itself
	 *
	 * @param other
	 *            other Neighbor whose sequenceIndexes need to be merged
	 */
	private void mergeSequenceIndexes(Neighbor other) {
		this.sequenceIndexes.addAll(other.sequenceIndexes);
	}

	/**
	 * Get hash code for the Point associated with object
	 *
	 * @return hash code for the Point
	 */
	private int getHashCodeForPoint() {
		return Arrays.hashCode(point);
	}

	private static class CollectorImpl
			implements
				Collector<Optional<Neighbor>, Map<Integer, Neighbor>, List<Neighbor>> {

		@Override
		public Supplier<Map<Integer, Neighbor>> supplier() {
			return HashMap::new;
		}

		@Override
		public BiConsumer<Map<Integer, Neighbor>, Optional<Neighbor>> accumulator() {
			return (neighborsMap, neighborOptional) -> {
				if (neighborOptional.isPresent()) {
					mergeNeighborIfNeededAndPut(neighborsMap, neighborOptional.get());
				}
			};
		}

		@Override
		public BinaryOperator<Map<Integer, Neighbor>> combiner() {
			return (left, right) -> {
				right.forEach((k, v) -> mergeNeighborIfNeededAndPut(left, v));
				return left;
			};
		}

		@Override
		public Function<Map<Integer, Neighbor>, List<Neighbor>> finisher() {
			return map -> {
				List<Neighbor> combinedResult = new ArrayList<>();
				map.forEach((k, v) -> {
					v.sequenceIndexes.sort(Long::compareTo);
					combinedResult.add(v);
				});
				Comparator<Neighbor> comparator = Comparator.comparingDouble(n -> n.distance);
				combinedResult.sort(comparator);
				return combinedResult;
			};
		}

		@Override
		public Set<Characteristics> characteristics() {
			return Collections.emptySet();
		}

		private void mergeNeighborIfNeededAndPut(Map<Integer, Neighbor> neighborsMap, Neighbor currentNeighbor) {
			Neighbor existingNeighbor = neighborsMap.get(currentNeighbor.getHashCodeForPoint());
			if (existingNeighbor != null) {
				existingNeighbor.mergeSequenceIndexes(currentNeighbor);
			} else {
				neighborsMap.put(currentNeighbor.getHashCodeForPoint(), currentNeighbor);
			}
		}
	}
}
