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

package com.amazon.randomcutforest.returntypes;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;

/**
 * A RangeVector is used when we want to track a quantity and its upper and
 * lower bounds
 */
public class RangeVector {

    public final float[] values;

    /**
     * An array of values corresponding to the upper ranges in each dimension.
     */
    public final float[] upper;
    /**
     * An array of values corresponding to the lower ranges in each dimension
     */
    public final float[] lower;

    public RangeVector(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        values = new float[dimensions];
        upper = new float[dimensions];
        lower = new float[dimensions];
    }

    /**
     * Construct a new RangeVector with the given number of spatial dimensions.
     * 
     * @param values the values being estimated in a range
     * @param upper  the higher values of the ranges
     * @param lower  the lower values in the ranges
     */
    public RangeVector(float[] values, float[] upper, float[] lower) {
        checkArgument(values.length > 0, " dimensions must be > 0");
        checkArgument(values.length == upper.length && upper.length == lower.length, "dimensions must be equal");
        for (int i = 0; i < values.length; i++) {
            checkArgument(upper[i] >= values[i] && values[i] >= lower[i], "incorrect semantics");
        }
        this.values = Arrays.copyOf(values, values.length);
        this.upper = Arrays.copyOf(upper, upper.length);
        this.lower = Arrays.copyOf(lower, lower.length);
    }

    public RangeVector(float[] values) {
        checkArgument(values.length > 0, "dimensions must be > 0 ");
        this.values = Arrays.copyOf(values, values.length);
        this.upper = Arrays.copyOf(values, values.length);
        this.lower = Arrays.copyOf(values, values.length);
    }

    /**
     * Create a deep copy of the base RangeVector.
     *
     * @param base The RangeVector to copy.
     */
    public RangeVector(RangeVector base) {
        int dimensions = base.values.length;
        this.values = Arrays.copyOf(base.values, dimensions);
        this.upper = Arrays.copyOf(base.upper, dimensions);
        this.lower = Arrays.copyOf(base.lower, dimensions);
    }

    public void shift(int i, float shift) {
        checkArgument(i >= 0 && i < values.length, "incorrect index");
        values[i] += shift;
        // managing precision
        upper[i] = max(values[i], upper[i] + shift);
        lower[i] = min(values[i], lower[i] + shift);
    }

    public void scale(int i, float weight) {
        checkArgument(i >= 0 && i < values.length, "incorrect index");
        checkArgument(weight > 0, " negative weight not permitted");
        values[i] = values[i] * weight;
        // managing precision
        upper[i] = max(upper[i] * weight, values[i]);
        lower[i] = min(lower[i] * weight, values[i]);
    }

}
