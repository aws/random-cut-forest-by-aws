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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

/**
 * A BoundingBox is an n-dimensional rectangle. Formally, for i = 1, ..., n
 * there are min and max values a_i and b_i, with a_i less than or equal to b_i,
 * such that the bounding box is equal to the set of points x whose ith
 * coordinate is between a_i and b_i.
 *
 * {@link Node}s in a {@link RandomCutTree} contain a BoundingBox, which is
 * always the smallest BoundingBox that contains all leaf points which are
 * descendents of the Node.
 */
public class BoundingBox extends AbstractBoundingBox<double[]> {

    public BoundingBox(double[] point) {
        super(point);
    }

    BoundingBox(final double[] minValues, final double[] maxValues, double sum) {
        super(minValues, maxValues, sum);
    }

    /**
     * creates a box out of the union of two points
     * 
     * @param first  first point
     * @param second second point
     */
    public BoundingBox(final double[] first, final double[] second) {
        super(new double[first.length], new double[second.length], 0.0);
        for (int i = 0; i < minValues.length; ++i) {
            minValues[i] = Math.min(first[i], second[i]);
            maxValues[i] = Math.max(first[i], second[i]);
            rangeSum += maxValues[i] - minValues[i];
        }

    }

    @Override
    public BoundingBox copy() {
        return new BoundingBox(Arrays.copyOf(minValues, minValues.length), Arrays.copyOf(maxValues, maxValues.length),
                rangeSum);
    }

    @Override
    public BoundingBox getMergedBox(double[] point) {
        checkArgument(point.length == minValues.length, "incorrect length");
        double[] minValuesMerged = new double[minValues.length];
        double[] maxValuesMerged = new double[minValues.length];
        double sum = 0.0;

        for (int i = 0; i < minValues.length; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], point[i]);
            maxValuesMerged[i] = Math.max(maxValues[i], point[i]);
            sum += maxValuesMerged[i] - minValuesMerged[i];
        }
        return new BoundingBox(minValuesMerged, maxValuesMerged, sum);
    }

    @Override
    public IBoundingBoxView getMergedBox(IBoundingBoxView otherBox) {
        double[] minValuesMerged = new double[minValues.length];
        double[] maxValuesMerged = new double[minValues.length];
        double sum = 0.0;

        for (int i = 0; i < minValues.length; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], otherBox.getMinValue(i));
            maxValuesMerged[i] = Math.max(maxValues[i], otherBox.getMaxValue(i));
            sum += maxValuesMerged[i] - minValuesMerged[i];
        }
        return new BoundingBox(minValuesMerged, maxValuesMerged, sum);
    }

    @Override
    public BoundingBox getMergedBox(AbstractBoundingBox<double[]> otherBox) {
        double[] minValuesMerged = new double[minValues.length];
        double[] maxValuesMerged = new double[minValues.length];
        double sum = 0.0;

        for (int i = 0; i < minValues.length; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], otherBox.minValues[i]);
            maxValuesMerged[i] = Math.max(maxValues[i], otherBox.maxValues[i]);
            sum += maxValuesMerged[i] - minValuesMerged[i];
        }
        return new BoundingBox(minValuesMerged, maxValuesMerged, sum);
    }

    @Override
    public BoundingBox addPoint(double[] point) {
        checkArgument(minValues.length == point.length, "incorrect length");
        if (maxValues == minValues) {
            return new BoundingBox(minValues, point);
        }
        rangeSum = 0;
        for (int i = 0; i < point.length; ++i) {
            minValues[i] = Math.min(minValues[i], point[i]);
            maxValues[i] = Math.max(maxValues[i], point[i]);
            rangeSum += maxValues[i] - minValues[i];
        }
        return this;
    }

    @Override
    public int getDimensions() {
        return minValues.length;
    }

    public double getMaxValue(final int dimension) {
        return maxValues[dimension];
    }

    public double getMinValue(final int dimension) {
        return minValues[dimension];
    }

    /**
     * Gets the range for a given dimensions.
     *
     * @param dimension for which we need the range
     * @return the range for the specified dimension
     */
    public double getRange(final int dimension) {
        return maxValues[dimension] - minValues[dimension];
    }

    @Override
    public String toString() {
        return String.format("BoundingBox(%s, %s)", Arrays.toString(minValues), Arrays.toString(maxValues));
    }

    @Override
    public boolean contains(double[] point) {
        checkArgument(point.length == getDimensions(), " incorrect lengths");
        for (int i = 0; i < point.length; i++) {
            if (minValues[i] > point[i] || maxValues[i] < point[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Two bounding boxes are considered equal if they have the same dimensions and
     * all their min values and max values are the same. Min and max values are
     * compared as primitive doubles using ==, so two bounding boxes are not equal
     * if their min and max values are merely very close.
     *
     * @param other An object to test for equality
     * @return true if other is a bounding box with the same min and max values
     */
    @Override
    public boolean equals(Object other) {
        if (!(other instanceof BoundingBox)) {
            return false;
        }

        AbstractBoundingBox<double[]> otherBox = (BoundingBox) other;
        return Arrays.equals(minValues, otherBox.minValues) && Arrays.equals(maxValues, otherBox.maxValues);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(minValues) + 31 * Arrays.hashCode(maxValues);
    }
}
