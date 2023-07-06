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

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.util.Weighted;

public class MultiCenter extends GenericMultiCenter<float[]> {

    ArrayList<Weighted<Integer>> assignedPoints;

    MultiCenter(float[] coordinate, float weight, double shrinkage, int numberOfRepresentatives) {
        super(coordinate, weight, shrinkage, numberOfRepresentatives);
        this.assignedPoints = new ArrayList<>();
    }

    public static MultiCenter initialize(float[] coordinate, float weight, double shrinkage,
            int numberOfRepresentatives) {
        checkArgument(shrinkage >= 0 && shrinkage <= 1.0, " parameter has to be in [0,1]");
        checkArgument(numberOfRepresentatives > 0 && numberOfRepresentatives <= 100,
                " the number of representatives has to be in (0,100]");
        return new MultiCenter(coordinate, weight, shrinkage, numberOfRepresentatives);
    }

    public void addPoint(int index, float weight, double dist, float[] point,
            BiFunction<float[], float[], Double> distance) {
        super.addPoint(index, weight, dist, point, distance);
        assignedPoints.add(new Weighted<>(index, weight));
    }

    // the following sets up reassignment of the coordinate based on the points
    // assigned to the center
    public void reset() {
        super.reset();
        assignedPoints = new ArrayList<>();
    }

    // a standard reassignment using the median values and NOT the mean; the mean is
    // unlikely to
    // provide robust convergence
    public double recompute(Function<Integer, float[]> getPoint, boolean force,
            BiFunction<float[], float[], Double> distanceFunction) {
        if (assignedPoints.size() == 0 || weight == 0.0 || !force) {
            return 0;
        }

        previousSumOFRadius = sumOfRadius;
        sumOfRadius = 0;
        for (int j = 0; j < assignedPoints.size(); j++) {
            // distance will check for -negative internally
            double addTerm = distance(getPoint.apply(assignedPoints.get(j).index), distanceFunction)
                    * assignedPoints.get(j).weight;
            sumOfRadius += addTerm;
        }
        return (previousSumOFRadius - sumOfRadius);

    }

    @Override
    public List<Weighted<Integer>> getAssignedPoints() {
        return assignedPoints;
    }
}
