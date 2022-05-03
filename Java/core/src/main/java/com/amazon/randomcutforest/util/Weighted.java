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

package com.amazon.randomcutforest.util;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * a container class that manages weights
 * 
 * @param <Q>
 */
public class Weighted<Q> {

    public Q index;
    public float weight;

    public Weighted(Q object, float weight) {
        this.index = object;
        this.weight = weight;
    }

    /**
     * a generic MonteCarlo sampler that creates an Arraylist of WeightedIndexes
     * 
     * @param input               input list of weighted objects
     * @param seed                random seed for repreoducibility
     * @param forceSampleFraction add the items which have weight over this fraction
     * @param scale               scale that multiples the weights of the remainder.
     *                            Note that elements that are sampled are rescaled
     *                            to have ensured that the total weight (after
     *                            removal of heavy items) remains the same in
     *                            expectation
     * @param <Q>                 a generic index type, typically float[] in the
     *                            current usage
     * @return a randomly sampled arraylist (which can be the same list) of length
     *         about LengthBound
     */
    public static <Q> List<Weighted<Q>> createSample(List<Weighted<Q>> input, long seed, int lengthBound,
            double forceSampleFraction, double scale) {

        if (input.size() < lengthBound) {
            return input;
        }

        ArrayList<Weighted<Q>> samples = new ArrayList<>();
        Random rng = new Random(seed);
        double totalWeight = input.stream().map(x -> (double) x.weight).reduce(Double::sum).get();
        double remainder = totalWeight;

        if (forceSampleFraction > 0) {
            remainder = input.stream().map(e -> {
                if (e.weight > totalWeight * forceSampleFraction) {
                    samples.add(new Weighted<>(e.index, e.weight));
                    return 0.0;
                } else {
                    return (double) e.weight;
                }
            }).reduce(Double::sum).get();
        }
        float factor = (float) (lengthBound * 1.0 / input.size());
        float newScale = (float) (scale * (remainder / totalWeight) / factor);
        input.stream().forEach(e -> {
            if ((e.weight <= totalWeight * forceSampleFraction) && (rng.nextDouble() < factor)) {
                samples.add(new Weighted<>(e.index, e.weight * newScale));
            }
        });

        return samples;
    }

    /**
     * an utility routine to pick the element such that the prefix sum including
     * that element exceeds a weight (or is the last element)
     * 
     * @param points a list of weighted objects
     * @param wt     a parameter determining the cumulative weight
     * @return the position of the item satisfying the prefix condition or the last
     *         element
     */

    public static <Q> Weighted<Q> prefixPick(List<Weighted<Q>> points, double wt) {
        checkArgument(points.size() > 0, "cannot pick from an empty list");
        double running = wt;
        Weighted<Q> saved = points.get(0);
        for (Weighted<Q> point : points) {
            if (running - point.weight <= 0.0) {
                return point;
            }
            running -= point.weight;
            saved = point;
        }
        return saved;
    }
}
