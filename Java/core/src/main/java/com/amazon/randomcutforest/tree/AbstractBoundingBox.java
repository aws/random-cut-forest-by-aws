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

/**
 * A BoundingBox is an n-dimensional rectangle. Formally, for i = 1, ..., n
 * there are min and max values a_i and b_i, with a_i less than or equal to b_i,
 * such that the bounding box is equal to the set of points x whose ith
 * coordinate is between a_i and b_i. Thus topologically we need the two corners
 * which define a box. While the current library considers specific realized
 * boxes, these boxes may correspond to implicit representations.
 *
 */
public abstract class AbstractBoundingBox<Point> implements IBoundingBoxView {

    /**
     * An array containing the minimum value corresponding to each dimension.
     */
    protected final Point minValues;

    /**
     * An array containing the maximum value corresponding to each dimensions
     */
    protected final Point maxValues;

    /**
     * The sum of side lengths defined by this bounding box.
     */
    protected double rangeSum;

    /**
     * Creates a degenerate bounding box containing a single point.
     *
     * @param point the point for which we need a bounding box
     */
    public AbstractBoundingBox(Point point) {
        minValues = maxValues = point;
        // a copy in not needed because mergedBox would create a copy
        // addPoint, addBox would also create copies
        rangeSum = 0.0;
    }

    /**
     * Create a new BoundingBox with the given minimum values and maximum values.
     *
     * @param minValues The minimum values for each coordinate.
     * @param maxValues The maximum values for each coordinate
     */
    protected AbstractBoundingBox(final Point minValues, final Point maxValues, double sum) {
        this.minValues = minValues;
        this.maxValues = maxValues;
        rangeSum = sum;
    }

    public abstract AbstractBoundingBox<Point> copy();

    /**
     * Return a new bounding box which is the smallest bounding box that contains
     * this bounding box and otherBoundingBox.
     *
     * @param otherBox the bounding box being merged with this box
     * @return the smallest bounding box that contains this bounding box and
     *         otherBoundingBox;
     */
    public abstract IBoundingBoxView getMergedBox(IBoundingBoxView otherBox);

    /**
     * The following will perform merge in place; unless the current box is a point;
     * in which case it would produce a new box
     * 
     * @param point to be added to this box
     * @return merged bounding box
     */
    public abstract AbstractBoundingBox<Point> addPoint(final Point point);

    /**
     * The following will perform merge in place;
     *
     * @param otherBox to be added to this box
     * @return merged bounding box
     */
    public abstract AbstractBoundingBox<Point> addBox(final AbstractBoundingBox<Point> otherBox);

    /**
     * @return dimensions of the box
     */
    public abstract int getDimensions();

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
    public abstract double getMaxValue(final int dimension);

    /**
     * Gets the min value of the specified dimension.
     *
     * @param dimension the dimension for which we need the min value
     * @return the min value of the specified dimension
     */
    public abstract double getMinValue(final int dimension);

    /**
     * Returns true if the given point is contained in this bounding box. This is
     * equivalent to the point being a member of the set defined by this bounding
     * box.
     *
     * @param point with which we're performing the comparison
     * @return whether the point is contained by the bounding box
     */
    public boolean contains(double[] point) {
        checkArgument(point.length == getDimensions(), " incorrect lengths");
        for (int i = 0; i < point.length; i++) {
            if (!contains(i, point[i])) {
                return false;
            }
        }

        return true;
    }

    public abstract boolean contains(Point point);

    /**
     * Returns true if the given bounding box is contained inside this bounding box.
     * Equivalently, if the given bounding box is a subset of this bounding box.
     *
     * @param other Another bounding box that we are comparing to this bounding box.
     * @return true if the given bounding box is contained inside this bounding box,
     *         false otherwise.
     */
    public boolean contains(AbstractBoundingBox<?> other) {
        checkArgument(getDimensions() == other.getDimensions(), "incorrect dimensions");
        for (int i = 0; i < getDimensions(); i++) {
            if (!contains(i, other.getMinValue(i)) || !contains(i, other.getMaxValue(i))) {
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
    public boolean contains(int dimension, double value) {
        return getMaxValue(dimension) >= value && value >= getMinValue(dimension);
    }

    /**
     * Gets the range for a given dimensions.
     *
     * @param dimension for which we need the range
     * @return the range for the specified dimension
     */
    public abstract double getRange(final int dimension);

    @Override
    public abstract String toString();

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
    public abstract boolean equals(Object other);

    @Override
    public abstract int hashCode();
}
