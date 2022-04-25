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

/**
 * at the current moment the following behaves as a weighted-point, however
 * "Weighted-X" has been used elsewhere in the library and in the fullness of
 * time, this class can evolve.
 */
public class ProjectedPoint {

    // basic
    float[] coordinate;

    // weight of the point, can be 1.0 for input point
    float weight;

    public ProjectedPoint(float[] coordinate, float weight) {
        // the following is to avoid copies of points as they are being moved
        // these values should NOT be altered
        this.coordinate = coordinate;
        this.weight = weight;
    }

    public int getDimension() {
        return coordinate.length;
    }

    public float getWeight() {
        return weight;
    }

    public float getCoordinate(int i) {
        return coordinate[i];
    }

    public void accumulateForDeviation(double[] sumArray, double[] squareSumArray) {
        for (int i = 0; i < coordinate.length; i++) {
            checkArgument(!Float.isNaN(weight) && Float.isFinite(weight),
                    " improper weight, accumulation not meaningful");
            checkArgument(!Float.isNaN(coordinate[i]) && Float.isFinite(coordinate[i]),
                    " improper input, in coordinate " + i + ", accumulation is not meaningful");
            sumArray[i] = coordinate[i] * weight;
            squareSumArray[i] = coordinate[i] * coordinate[i] * weight;
        }
    }
}
