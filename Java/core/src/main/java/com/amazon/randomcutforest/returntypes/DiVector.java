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
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import com.amazon.randomcutforest.anomalydetection.AnomalyAttributionVisitor;
import java.util.Arrays;
import java.util.function.Function;

/**
 * A DiVector is used when we want to track a quantity in both the positive and negative directions
 * for each dimension in a manifold. For example, when using a {@link AnomalyAttributionVisitor} to
 * compute the attribution of the anomaly score to dimension of the input point, we want to know if
 * the anomaly score attributed to the ith coordinate of the input point is due to that coordinate
 * being unusually high or unusually low.
 */
public class DiVector {

    /** An array of values corresponding to the positive direction in each dimension. */
    public final double[] high;
    /** An array of values corresponding to the negative direction in each dimension. */
    public final double[] low;

    private final int dimensions;

    /**
     * Construct a new DiVector with the given number of spatial dimensions. In the result, {@link
     * #high} and {@link #low} will each contain this many variates.
     *
     * @param dimensions The number of dimensions of data to store.
     */
    public DiVector(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        high = new double[dimensions];
        low = new double[dimensions];
    }

    /**
     * Create a deep copy of the base DiVector.
     *
     * @param base The DiVector to copy.
     */
    public DiVector(DiVector base) {
        this.dimensions = base.dimensions;
        high = Arrays.copyOf(base.high, dimensions);
        low = Arrays.copyOf(base.low, dimensions);
    }

    /**
     * Add the values of {@link #high} and {@link #low} from the right vector to the left vector and
     * return the left vector. This method is used to accumulate DiVector results.
     *
     * @param left The DiVector we are modifying. After calling this method, the low and high values
     *     in the DiVector will contain a sum of the previous values and the corresponding values
     *     from the right vector.
     * @param right A DiVector that we want to add to the left vector. This DiVector is not modified
     *     by the method.
     * @return the modified left vector.
     */
    public static DiVector addToLeft(DiVector left, DiVector right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");

        for (int i = 0; i < left.dimensions; i++) {
            left.high[i] += right.high[i];
            left.low[i] += right.low[i];
        }

        return left;
    }

    /** @return the number of spatial dimensions of this DiVector. */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * Return a new DiVector where each value in high and low is equal to z times the corresponding
     * value in this DiVector.
     *
     * @param z The scaling factor.
     * @return a new DiVector where each value in high and low is equal to z times the corresponding
     *     value in this DiVector.
     */
    public DiVector scale(double z) {
        DiVector result = new DiVector(dimensions);
        for (int i = 0; i < dimensions; i++) {
            result.high[i] = high[i] * z;
            result.low[i] = low[i] * z;
        }
        return result;
    }

    /**
     * If the L1 norm of this DiVector is positive, scale the values in high and low so that the new
     * L1 norm is equal to the target value. If the current L1 norm is 0, do nothing.
     *
     * @param targetNorm The target L1 norm value.
     */
    public void renormalize(double targetNorm) {
        double norm = getHighLowSum();
        if (norm > 0) {
            double scaleFactor = targetNorm / norm;
            for (int i = 0; i < dimensions; i++) {
                high[i] = high[i] * scaleFactor;
                low[i] = low[i] * scaleFactor;
            }
        }
    }

    /**
     * Apply the given function to each component of DiVector. That is, each entry of both the high
     * and low arrays is transformed using this function.
     *
     * @param function A function to apply to every entry of the high and low arrays in this
     *     DiVector.
     */
    public void componentwiseTransform(Function<Double, Double> function) {
        for (int i = 0; i < dimensions; i++) {
            high[i] = function.apply(high[i]);
            low[i] = function.apply(low[i]);
        }
    }

    /**
     * Return the sum of high and low in the ith coordinate.
     *
     * @param i A coordinate index
     * @return the sum of high and low in the ith coordinate.
     */
    public double getHighLowSum(int i) {
        return high[i] + low[i];
    }

    /** @return the sum of all values in the high and low arrays. */
    public double getHighLowSum() {
        double score = 0.0;
        for (int i = 0; i < dimensions; i++) {
            score += high[i] + low[i];
        }
        return score;
    }
}
