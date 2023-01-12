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

package com.amazon.randomcutforest.summarization;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.util.Weighted;

/**
 * the following class abstracts a single centroid representation of a group of
 * points. The class is modeled after the well scattered representatives used in
 * CURE https://en.wikipedia.org/wiki/CURE_algorithm
 *
 * The number of representatives (refered as c in above) determines the possible
 * shapes that can be represented. Setting c=1 corresponds to stnadard centroid
 * based clustering
 *
 * The parameter shrinkage is slightly different from its usage in CURE,
 * although the idea of its use is similar. The main reason is that CURE was
 * designed for geometric spaces, and RCFSummarize is designed to support
 * arbitrary distance based clustering; once the user provides a distance
 * function from (R, R) into double based on ideas of STREAM
 * https://en.wikipedia.org/wiki/Data_stream_clustering In CURE, shrinkage was
 * used to create representatives close to the center of a cluster which is
 * impossible for generic types R. Instead shrinkage value in [0,1] corresponds
 * to morphing the distance function to "pretend" as if the distance is to the
 * primary representative of the cluster.
 *
 * This generic version does not store any assigned points. As a result the size
 * is bounded and these clusters are ideal for streaming algorithms where
 * resource usage would not increase with more data.
 */
public class GenericMultiCenter<R> implements ICluster<R> {

    public static int DEFAULT_NUMBER_OF_REPRESENTATIVES = 5;
    public static double DEFAULT_SHRINKAGE = 0.0;
    int numberOfRepresentatives = DEFAULT_NUMBER_OF_REPRESENTATIVES;
    double shrinkage = DEFAULT_SHRINKAGE;

    ArrayList<Weighted<R>> representatives;
    double weight;
    double sumOfRadius;

    double previousWeight = 0;
    double previousSumOFRadius = 0;

    GenericMultiCenter(R coordinate, float weight, double shrinkage, int numberOfRepresentatives) {
        // explicitly copied because array elements will change
        this.representatives = new ArrayList<>();
        this.representatives.add(new Weighted<>(coordinate, weight));
        this.weight = weight;
        this.numberOfRepresentatives = numberOfRepresentatives;
        this.shrinkage = shrinkage;
    }

    public static <R> GenericMultiCenter<R> initialize(R coordinate, float weight) {
        return new GenericMultiCenter<>(coordinate, weight, DEFAULT_SHRINKAGE, DEFAULT_NUMBER_OF_REPRESENTATIVES);
    }

    public static <R> GenericMultiCenter<R> initialize(R coordinate, float weight, double shrinkage,
            int numberOfRepresentatives) {
        checkArgument(shrinkage >= 0 && shrinkage <= 1.0, " parameter has to be in [0,1]");
        checkArgument(numberOfRepresentatives > 0 && numberOfRepresentatives <= 100,
                " the number of representatives has to be in (0,100]");
        return new GenericMultiCenter<>(coordinate, weight, shrinkage, numberOfRepresentatives);
    }

    // adds a point; only the index to keep space bounds lower
    // note that the weight may not be the entire weight of a point in case of a
    // "soft" assignment
    public void addPoint(int index, float weight, double dist, R point, BiFunction<R, R, Double> distance) {
        // accounting for the closest representative, if there are more than one
        Weighted<R> closest = representatives.get(0);
        if (representatives.size() > 1) {
            double newDist = distance.apply(point, representatives.get(0).index);
            for (int i = 1; i < representatives.size(); i++) {
                double t = distance.apply(point, representatives.get(i).index);
                if (t < newDist) {
                    newDist = t;
                    closest = representatives.get(i);
                }
            }
        }
        closest.weight += weight;
        this.weight += weight;
        this.sumOfRadius += weight * dist;
    }

    // the following sets up reassignment of the coordinate based on the points
    // assigned to the center
    public void reset() {
        previousWeight = weight;
        weight = 0;
        for (int i = 0; i < representatives.size(); i++) {
            representatives.get(i).weight = 0;
        }
        previousSumOFRadius = sumOfRadius;
        sumOfRadius = 0;
    }

    public double averageRadius() {
        return (weight > 0) ? sumOfRadius / weight : 0;
    }

    // forces a nearest neighbor merge
    public double extentMeasure() {
        return (weight > 0) ? 0.5 * sumOfRadius / (numberOfRepresentatives * weight) : 0;
    }

    public double getWeight() {
        return weight;
    }

    public boolean captureBeforeReset(R point, BiFunction<R, R, Double> distanceFunction) {
        return previousWeight * distance(point, distanceFunction) < 3 * previousSumOFRadius;
    }

    // reassignment may not be meaningful for generic types, without additional
    // information
    public double recompute(Function<Integer, R> getPoint, boolean flag, BiFunction<R, R, Double> distanceFunction) {
        return 0;
    }

    // merges a center into another
    public void absorb(ICluster<R> other, BiFunction<R, R, Double> distance) {
        List<Weighted<R>> savedRepresentatives = this.representatives;
        savedRepresentatives.addAll(other.getRepresentatives());
        this.representatives = new ArrayList<>();

        int maxIndex = 0;
        float weight = savedRepresentatives.get(0).weight;
        for (int i = 1; i < savedRepresentatives.size(); i++) {
            if (weight < savedRepresentatives.get(i).weight) {
                weight = savedRepresentatives.get(i).weight;
                maxIndex = i;
            }
        }
        this.representatives.add(savedRepresentatives.get(maxIndex));
        savedRepresentatives.remove(maxIndex);
        sumOfRadius += other.extentMeasure() * other.getWeight();
        this.weight += other.getWeight();

        /**
         * create a list of representatives based on the farthest point method, which
         * correspond to a well scattered set. See
         * https://en.wikipedia.org/wiki/CURE_algorithm
         */
        while (savedRepresentatives.size() > 0 && this.representatives.size() < numberOfRepresentatives) {
            double farthestWeightedDistance = 0.0;
            int farthestIndex = Integer.MAX_VALUE;
            for (int j = 0; j < savedRepresentatives.size(); j++) {
                if (savedRepresentatives.get(j).weight > weight / (2 * numberOfRepresentatives)) {
                    double newWeightedDist = distance.apply(this.representatives.get(0).index,
                            savedRepresentatives.get(j).index) * savedRepresentatives.get(j).weight;
                    checkArgument(newWeightedDist >= 0, " weights or distances cannot be negative");
                    for (int i = 1; i < this.representatives.size(); i++) {
                        newWeightedDist = min(newWeightedDist,
                                distance.apply(this.representatives.get(i).index, savedRepresentatives.get(j).index))
                                * savedRepresentatives.get(j).weight;
                        checkArgument(newWeightedDist >= 0, " weights or distances cannot be negative");
                    }
                    if (newWeightedDist > farthestWeightedDistance) {
                        farthestWeightedDistance = newWeightedDist;
                        farthestIndex = j;
                    }
                }
            }
            if (farthestWeightedDistance == 0.0) {
                break;
            }
            this.representatives.add(savedRepresentatives.get(farthestIndex));
            savedRepresentatives.remove(farthestIndex);
        }

        // absorb the remainder into existing represen tatives
        for (Weighted<R> representative : savedRepresentatives) {
            double dist = distance.apply(representative.index, this.representatives.get(0).index);
            checkArgument(dist >= 0, "distance cannot be negative");
            double minDist = dist;
            int minIndex = 0;
            for (int i = 1; i < this.representatives.size(); i++) {
                double newDist = distance.apply(this.representatives.get(i).index, representative.index);
                checkArgument(newDist >= 0, "distance cannot be negative");
                if (newDist < minDist) {
                    minDist = newDist;
                    minIndex = i;
                }
            }
            this.representatives.get(minIndex).weight += representative.weight;
            sumOfRadius += representative.weight * ((1 - shrinkage) * minDist + dist * shrinkage);
        }
    }

    @Override
    public double distance(R point, BiFunction<R, R, Double> distanceFunction) {
        double dist = distanceFunction.apply(this.representatives.get(0).index, point);
        checkArgument(dist >= 0, "distance cannot be negative");
        double newDist = dist;
        for (int i = 1; i < this.representatives.size(); i++) {
            newDist = min(newDist, distanceFunction.apply(this.representatives.get(i).index, point));
            checkArgument(newDist >= 0, "distance cannot be negative");
        }
        return (1 - shrinkage) * newDist + shrinkage * dist;
    }

    @Override
    public double distance(ICluster<R> other, BiFunction<R, R, Double> distanceFunction) {
        List<Weighted<R>> representatives = other.getRepresentatives();
        double dist = distanceFunction.apply(this.representatives.get(0).index, representatives.get(0).index);
        checkArgument(dist >= 0, "distance cannot be negative");
        double newDist = dist;
        for (int i = 1; i < this.representatives.size(); i++) {
            for (int j = 1; j < representatives.size(); j++) {
                newDist = min(newDist,
                        distanceFunction.apply(this.representatives.get(i).index, representatives.get(j).index));
                checkArgument(newDist >= 0, "distance cannot be negative");
            }
        }
        return (1 - shrinkage) * newDist + shrinkage * dist;
    }

    @Override
    public List<Weighted<R>> getRepresentatives() {
        return representatives;
    }

}
