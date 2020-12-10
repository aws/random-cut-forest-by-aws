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
public class BoundingBox implements IBoundingBox<double[]> {

    /**
     * An array containing the minimum value corresponding to each dimension.
     */
    private final double[] minValues;

    /**
     * An array containing the maximum value corresponding to each dimensions
     */
    private final double[] maxValues;

    /**
     * The number of dimensions which this bounding box describes.
     */
    private final int dimensions;

    /**
     * The sum of side lengths defined by this bounding box.
     */
    private double rangeSum;

    /**
     * Creates a degenerate bounding box containing a single point.
     *
     * @param point the point for which we need a bounding box
     */
    public BoundingBox(double[] point) {
        dimensions = point.length;
        minValues = maxValues = Arrays.copyOf(point, point.length);
        rangeSum = 0.0;
    }

    /**
     * Create a new BoundingBox with the given minimum values and maximum values.
     *
     * @param minValues The minimum values for each coordinate.
     * @param maxValues The maximum values for each coordinate
     */
    protected BoundingBox(final double[] minValues, final double[] maxValues) {
        this.minValues = minValues;
        this.maxValues = maxValues;
        double sum = 0.0;

        for (int i = 0; i < minValues.length; ++i) {
            sum += maxValues[i] - minValues[i];
        }

        rangeSum = sum;
        dimensions = minValues.length;
    }

    public int getDimensions() {
        return dimensions;
    }

    /**
     * Return a new bounding box which is the smallest bounding box that contains
     * this bounding box and otherBoundingBox.
     *
     * @param otherBoundingBox the bounding box being merged with this box
     * @return the smallest bounding box that contains this bounding box and
     *         otherBoundingBox;
     */
    public BoundingBox getMergedBox(final BoundingBox otherBoundingBox) {
        double[] minValuesMerged = new double[dimensions];
        double[] maxValuesMerged = new double[dimensions];

        for (int i = 0; i < dimensions; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], otherBoundingBox.minValues[i]);
            maxValuesMerged[i] = Math.max(maxValues[i], otherBoundingBox.maxValues[i]);
        }

        return new BoundingBox(minValuesMerged, maxValuesMerged);
    }

    @Override
    public BoundingBox getMergedBox(final IBoundingBox<double[]> otherBoundingBox) {
        return getMergedBox(otherBoundingBox.convertBoxToDouble());
    }

    /**
     * Return a new bounding box which is the smallest bounding box that contains
     * this bounding box and the given point.
     *
     * @param point the new point being added to the box
     * @return the smallest bounding box that contains this bounding box and the
     *         given point.
     */
    public BoundingBox getMergedBox(double[] point) {
        double[] minValuesMerged = new double[dimensions];
        double[] maxValuesMerged = new double[dimensions];

        for (int i = 0; i < dimensions; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], point[i]);
            maxValuesMerged[i] = Math.max(maxValues[i], point[i]);
        }

        return new BoundingBox(minValuesMerged, maxValuesMerged);
    }

    @Override
    public BoundingBox convertBoxToDouble() {
        return this;
    }

    @Override
    public BoundingBoxFloat convertBoxToFloat() {
        return null;
    }

    /**
     * Returns a bounding box of two points
     * 
     * @param point      the first point
     * @param otherPoint the second point
     * @return a bounding box that covers both points
     */
    public static BoundingBox getMergedBox(double[] point, double[] otherPoint) {
        double[] minValuesMerged = new double[point.length];
        double[] maxValuesMerged = new double[point.length];

        for (int i = 0; i < point.length; ++i) {
            minValuesMerged[i] = Math.min(otherPoint[i], point[i]);
            maxValuesMerged[i] = Math.max(otherPoint[i], point[i]);
        }
        return new BoundingBox(minValuesMerged, maxValuesMerged);
    }

    @Override
    public IBoundingBox<double[]> addPoint(double[] point) {
        rangeSum = 0;
        for (int i = 0; i < point.length; ++i) {
            minValues[i] = Math.min(minValues[i], point[i]);
            maxValues[i] = Math.max(maxValues[i], point[i]);
            rangeSum += maxValues[i] - minValues[i];
        }
        return this;
    }

    /**
     * @return the sum of side lengths for this BoundingBox.
     */
    public double getRangeSum() {
        return rangeSum;
    }

    /**
     * Gets the max value of the specified dimension.
     *
     * @param dimension the dimension for which we need the max value
     * @return the max value of the specified dimension
     */
    public double getMaxValue(final int dimension) {
        return maxValues[dimension];
    }

    /**
     * Gets the min value of the specified dimension.
     *
     * @param dimension the dimension for which we need the min value
     * @return the min value of the specified dimension
     */
    public double getMinValue(final int dimension) {
        return minValues[dimension];
    }

    /**
     * Returns true if the given point is contained in this bounding box. This is
     * equivalent to the point being a member of the set defined by this bounding
     * box.
     *
     * @param point with which we're performing the comparison
     * @return whether the point is contained by the bounding box
     */
    public boolean contains(double[] point) {
        for (int i = 0; i < dimensions; i++) {
            if (!contains(i, point[i])) {
                return false;
            }
        }

        return true;
    }

    /**
     * Returns true if the given bounding box is contained inside this bounding box.
     * Equivalently, if the given bounding box is a subset of this bounding box.
     *
     * @param other Another bounding box that we are comparing to this bounding box.
     * @return true if the given bounding box is contained inside this bounding box,
     *         false otherwise.
     */
    public boolean contains(BoundingBox other) {
        for (int i = 0; i < dimensions; i++) {
            if (!contains(i, other.minValues[i]) || !contains(i, other.maxValues[i])) {
                return false;
            }
        }

        return true;
    }

    /**
     * Test whether a given scalar value falls between the min and max values in the
     * given dimension.
     *
     * @return whether the value of a point is between the minimum or maximum value
     *         of the bounding box for the given dimension
     */
    private boolean contains(int dimension, double value) {
        return maxValues[dimension] >= value && value >= minValues[dimension];
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

        BoundingBox otherBox = (BoundingBox) other;
        return Arrays.equals(minValues, otherBox.minValues) && Arrays.equals(maxValues, otherBox.maxValues);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(minValues) + 31 * Arrays.hashCode(maxValues);
    }
}
