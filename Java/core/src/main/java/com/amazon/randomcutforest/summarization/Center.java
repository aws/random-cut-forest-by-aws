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

import static java.lang.Math.exp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.util.Weighted;

/**
 * the following class abstracts a single centroid representation of a group of
 * points
 */
public class Center implements ICluster {

    float[] representative;
    double weight;
    ArrayList<Weighted<Integer>> assignedPoints;
    double sumOfRadius;

    double previousWeight = 0;
    double previousSumOFRadius = 0;

    Center(float[] coordinate, float weight) {
        // explicitly copied because array elements will change
        this.representative = Arrays.copyOf(coordinate, coordinate.length);
        this.weight = weight;
        this.assignedPoints = new ArrayList<>();
    }

    public static Center initialize(float[] coordinate, float weight) {
        return new Center(coordinate, weight);
    }

    // adds a point; only the index to keep space bounds lower
    // note that the weight may not be the entire weight of a point in case of a
    // "soft" assignment
    public void addPoint(int index, float weight, double dist) {
        assignedPoints.add(new Weighted<>(index, weight));
        this.weight += weight;
        this.sumOfRadius += weight * dist;
    }

    // the following sets up reassignment of the coordinate based on the points
    // assigned to the center
    public void reset() {
        assignedPoints = new ArrayList<>();
        previousWeight = weight;
        weight = 0;
        previousSumOFRadius = sumOfRadius;
    }

    // average radius computation
    public double averageRadius() {
        return (weight > 0) ? sumOfRadius / weight : 0;
    }

    public double getWeight() {
        return weight;
    }

    public boolean captureBeforeReset(float[] point, BiFunction<float[], float[], Double> distance) {
        return previousWeight * distance.apply(point, representative) < 3 * previousSumOFRadius;
    }

    // a standard reassignment using the median values and NOT the mean; the mean is
    // unlikely to
    // provide robust convergence
    public double recompute(List<Weighted<float[]>> points, BiFunction<float[], float[], Double> distance) {
        if (assignedPoints.size() == 0 || weight == 0.0) {
            Arrays.fill(representative, 0); // zero out values
            return 0;
        }

        previousSumOFRadius = sumOfRadius;
        sumOfRadius = 0;
        for (int i = 0; i < representative.length; i++) {
            int index = i;
            // the following would be significantly slow unless points are backed by arrays
            assignedPoints.sort(
                    (o1, o2) -> Double.compare(points.get(o1.index).index[index], points.get(o2.index).index[index]));
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
            representative[index] = points.get(assignedPoints.get(position).index).index[index];
        }
        for (int j = 0; j < assignedPoints.size(); j++) {
            sumOfRadius += distance.apply(representative, points.get(assignedPoints.get(j).index).index)
                    * assignedPoints.get(j).weight;
        }
        return (previousSumOFRadius - sumOfRadius);

    }

    // merges a center into another
    // this can be followed by a reassignment step; however the merger uses a
    // sigmoid based weightage
    // for robustness
    public void absorb(ICluster other, BiFunction<float[], float[], Double> distance) {
        List<Weighted<float[]>> representatives = other.getRepresentatives();
        float[] closest = representatives.get(0).index;
        double dist = Double.MAX_VALUE;
        for (Weighted<float[]> e : representatives) {
            double t = distance.apply(e.index, representative);
            if (t < dist) {
                dist = t;
                closest = e.index;
            }
        }

        double otherWeight = other.getWeight();
        double expRatio = exp(2 * (weight - otherWeight) / (weight + otherWeight));
        double factor = expRatio / (1.0 + expRatio);
        for (int i = 0; i < representative.length; i++) {
            representative[i] = (float) (factor * representative[i] + (1 - factor) * closest[i]);
        }
        // distance is (approximately) the reverse of the ratio
        // this computation is meant to be approximate
        sumOfRadius += (weight * (1.0 - factor) + otherWeight * factor) * dist;
        weight += otherWeight;
    }

    public double distance(float[] point, BiFunction<float[], float[], Double> distance) {
        return distance.apply(point, representative);
    }

    @Override
    public double distance(ICluster other, BiFunction<float[], float[], Double> distance) {
        return other.distance(representative, distance);
    }

    @Override
    public float[] primaryRepresentative(BiFunction<float[], float[], Double> distance) {
        return Arrays.copyOf(representative, representative.length);
    }

    @Override
    public List<Weighted<float[]>> getRepresentatives() {
        ArrayList<Weighted<float[]>> answer = new ArrayList<>();
        answer.add(new Weighted<>(representative, (float) weight));
        return answer;
    }

}
