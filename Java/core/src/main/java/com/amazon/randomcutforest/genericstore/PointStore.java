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

package com.amazon.randomcutforest.genericstore;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class PointStore extends RCStore<double[]> {

    private final int dimensions;

    public PointStore(int dimensions, int capacity) {
        super(double[].class, capacity);
        this.dimensions = dimensions;
    }

    @Override
    public int add(double[] point) {
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");
        return super.add(point);
    }

    public boolean pointEquals(int index, double[] point) {
        checkValidIndex(index);
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");

        double[] storedPoint = get(index);
        for (int j = 0; j < dimensions; j++) {
            if (point[j] != storedPoint[j]) {
                return false;
            }
        }

        return true;
    }
}
