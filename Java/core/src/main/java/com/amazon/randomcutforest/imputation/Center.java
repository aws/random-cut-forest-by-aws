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

package com.amazon.randomcutforest.imputation;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.exp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

/**
 * the following class abstracts a single centroid representation of a group of
 * points
 */
public class Center {
    /**
     * a factor that controls weight assignment for soft clustering; this is the
     * multiple of the minimum distance and should be greater or equal 1.
     */
    public static double WEIGHT_ALLOCATION_THRESHOLD = 1.25;

    float[] coordinate;
    double weight;
    ArrayList<AssignedPoint> assignedPoints;
    double sumOfRadius;

    double previousWeight = 0;
    double previousSumOFRadius = 0;

    Center(float[] coordinate, float weight) {
        // explicitly copied because array elements will change
        this.coordinate = Arrays.copyOf(coordinate, coordinate.length);
        this.weight = weight;
        this.assignedPoints = new ArrayList<>();
    }

    // adds a point; only the index to keep space bounds lower
    // note that the weight may not be the entire weight of a point in case of a
    // "soft" assignment
    protected void add(int index, float weight) {
        assignedPoints.add(new AssignedPoint(index, weight));
        this.weight += weight;
    }

    // the following sets up reassignment of the coordinate based on the points
    // assigned to the center
    protected void reset() {
        assignedPoints = new ArrayList<>();
        previousWeight = weight;
        weight = 0;
        previousSumOFRadius = sumOfRadius;
    }

    // average radius computation
    public double radius() {
        return (weight > 0) ? sumOfRadius / weight : 0;
    }

    // a standard reassignment using the median values and NOT the mean; the mean is
    // unlikely to
    // provide robust convergence
    public void recompute(List<ProjectedPoint> points, BiFunction<float[], float[], Double> distance) {
        if (assignedPoints.size() == 0 || weight == 0.0) {
            Arrays.fill(coordinate, 0); // zero out values
            return;
        }

        sumOfRadius = 0;
        for (int i = 0; i < coordinate.length; i++) {
            int index = i;
            assignedPoints.sort((o1, o2) -> Double.compare(points.get(o1.index).coordinate[index],
                    points.get(o2.index).coordinate[index]));
            double runningWeight = weight / 2;
            int position = 0;
            while (runningWeight >= 0 && position < assignedPoints.size()) {
                if (runningWeight > assignedPoints.get(position).weight) {
                    runningWeight -= assignedPoints.get(position).weight;
                    ++position;
                } else {
                    break;
                }
            }
            coordinate[index] = points.get(assignedPoints.get(position).index).coordinate[index];
        }
        for (int j = 0; j < assignedPoints.size(); j++) {
            sumOfRadius += distance.apply(coordinate, points.get(assignedPoints.get(j).index).coordinate)
                    * assignedPoints.get(j).weight;
        }

    }

    // merges a center into another
    // this can be followed by a reassignment step; however the merger uses a
    // sigmoid based weightage
    // for robustness
    public void mergeInto(Center other, BiFunction<float[], float[], Double> distance) {
        double dist = distance.apply(coordinate, other.coordinate);
        double expRatio = exp(2 * (weight - other.weight) / (weight + other.weight));
        double factor = expRatio / (1.0 + expRatio);
        for (int i = 0; i < coordinate.length; i++) {
            coordinate[i] = (float) (factor * coordinate[i] + (1 - factor) * other.coordinate[i]);
        }
        // distance is (approximately) the reverse of the ratio
        // this computation is meant to be approximate
        sumOfRadius += (weight * (1.0 - factor) + other.weight * factor) * dist;
        weight += other.weight;
        other.weight = 0;
    }

    /**
     * an assignment of points to centers, with the ability to increase the number
     * of centers on the fly for typical usage, this capability (of increasing the
     * number of centers) should not be used because the operation corresponds to
     * the order of the input. This capability can be extremely handy for streams
     * (sampledPoints) with random order
     * 
     * @param limit         the maximum number of centers feasible(often this is
     *                      centers.size() from the callee)
     * @param sampledPoints the list of points
     * @param centers       the current list of centers
     * @param distance      a distance function
     */
    public static void assign(int limit, List<ProjectedPoint> sampledPoints, List<Center> centers,
            BiFunction<float[], float[], Double> distance) {
        checkArgument(limit > 0, " target size of centers cannot be 0");
        checkArgument(limit >= centers.size(), " limit cannot be below current size of centers");

        for (int i = 0; i < centers.size(); i++) {
            centers.get(i).reset();
        }

        for (int j = 0; j < sampledPoints.size(); j++) {
            if (sampledPoints.get(j).weight > 0) {
                double[] dist = new double[centers.size()];
                Arrays.fill(dist, Double.MAX_VALUE);
                double minDist = Double.MAX_VALUE;
                int minDistNbr = -1;
                boolean captured = false;
                for (int i = 0; i < centers.size(); i++) {
                    dist[i] = distance.apply(sampledPoints.get(j).coordinate, centers.get(i).coordinate);
                    if (minDist > dist[i]) {
                        minDist = dist[i];
                        minDistNbr = i;
                    }
                    // the following is necessary to maintain the invariant that centers do not have
                    // distance 0
                    if (minDist == 0) {
                        break;
                    }
                    if (dist[i] * centers.get(i).previousWeight < 3 * centers.get(i).previousSumOFRadius) {
                        captured = true;
                    }
                }

                if (minDist == 0) {
                    centers.get(minDistNbr).add(j, sampledPoints.get(j).weight);
                } else if (!captured && centers.size() < limit) {
                    Center c = new Center(sampledPoints.get(j).coordinate, sampledPoints.get(j).weight);
                    c.add(j, sampledPoints.get(j).weight);
                    centers.add(c);
                } else {
                    double sum = 0;
                    for (int i = 0; i < centers.size(); i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            sum += minDist / dist[i]; // setting up harmonic mean
                        }
                    }
                    for (int i = 0; i < centers.size(); i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            // harmonic mean
                            centers.get(i).add(j, (float) (sampledPoints.get(j).weight * minDist / (dist[i] * sum)));
                        }
                    }
                }
            }
        }

        for (int i = 0; i < centers.size(); i++) {
            centers.get(i).recompute(sampledPoints, distance);
        }
    }
}

/**
 * at the current moment the following behaves as a weighted-point, however
 * "Weighted-X" has been used elsewhere in the library and in the fullness of
 * time, this class can evolve.
 */
class AssignedPoint {

    // basic
    int index;

    // weight of the point, can be 1.0 for input point
    float weight;

    AssignedPoint(int index, float weight) {
        // the following is to avoid copies of points as they are being moved
        // these values should NOT be altered
        this.index = index;
        this.weight = weight;
    }
}
