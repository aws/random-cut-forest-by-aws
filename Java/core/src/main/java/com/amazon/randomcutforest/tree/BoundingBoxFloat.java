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
import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;

import java.util.Arrays;

/**
 * A single precision implementation of AbstractBoundingBox which also satisfies
 * the interface for Visitor classes
 */
public class BoundingBoxFloat extends AbstractBoundingBox<float[]> {

    public BoundingBoxFloat(float[] point) {
        super(point);
    }

    public BoundingBoxFloat(final float[] minValues, final float[] maxValues, double sum) {
        super(minValues, maxValues, sum);
    }

    public BoundingBoxFloat(final float[] first, final float[] second) {
        super(new float[first.length], new float[second.length], 0.0);
        for (int i = 0; i < minValues.length; ++i) {
            minValues[i] = Math.min(first[i], second[i]);
            maxValues[i] = Math.max(first[i], second[i]);
            rangeSum += maxValues[i] - minValues[i];
        }

    }

    @Override
    public AbstractBoundingBox<float[]> copy() {
        return new BoundingBoxFloat(Arrays.copyOf(minValues, minValues.length),
                Arrays.copyOf(maxValues, maxValues.length), rangeSum);
    }

    public BoundingBoxFloat newCopy() {
        return new BoundingBoxFloat(Arrays.copyOf(minValues, minValues.length),
                Arrays.copyOf(maxValues, maxValues.length), rangeSum);
    }

    @Override
    public IBoundingBoxView getMergedBox(double[] point) {
        return getMergedBox(toFloatArray(point));
    }

    @Override
    public IBoundingBoxView getMergedBox(IBoundingBoxView otherBox) {
        float[] minValuesMerged = new float[minValues.length];
        float[] maxValuesMerged = new float[minValues.length];
        double sum = 0.0;

        for (int i = 0; i < minValues.length; ++i) {
            minValuesMerged[i] = Math.min(minValues[i], (float) otherBox.getMinValue(i));
            maxValuesMerged[i] = Math.max(maxValues[i], (float) otherBox.getMaxValue(i));
            sum += maxValuesMerged[i] - minValuesMerged[i];
        }
        return new BoundingBoxFloat(minValuesMerged, maxValuesMerged, sum);
    }

    public void replaceBox(float[] point) {
        System.arraycopy(point, 0, minValues, 0, point.length);
        System.arraycopy(point, 0, maxValues, 0, point.length);
        rangeSum = 0;
    }

    public void copyFrom(BoundingBoxFloat otherBox) {
        System.arraycopy(otherBox.minValues, 0, minValues, 0, otherBox.minValues.length);
        System.arraycopy(otherBox.maxValues, 0, maxValues, 0, otherBox.maxValues.length);
        rangeSum = otherBox.rangeSum;
    }

    public double probabilityOfCut(float[] point) {
        double range = 0;
        for (int i = 0; i < point.length; i++) {
            range += Math.max(minValues[i] - point[i], 0);
        }
        for (int i = 0; i < point.length; i++) {
            range += Math.max(point[i] - maxValues[i], 0);
        }
        if (range == 0) {
            return 0;
        } else if (rangeSum == 0) {
            return 1;
        } else {
            return range / (range + rangeSum);
        }
    }

    public IBoundingBoxView getMergedBox(float[] point) {
        checkArgument(point.length == minValues.length, "incorrect length");
        return copy().addPoint(point);
    }

    public float[] getMaxValues() {
        return maxValues;
    }

    public float[] getMinValues() {
        return minValues;
    }

    @Override
    public BoundingBoxFloat addPoint(float[] point) {
        checkArgument(minValues.length == point.length, "incorrect length");
        checkArgument(minValues != maxValues, "not a mutable box");
        rangeSum = 0;
        for (int i = 0; i < point.length; ++i) {
            minValues[i] = Math.min(minValues[i], point[i]);
        }
        for (int i = 0; i < point.length; ++i) {
            maxValues[i] = Math.max(maxValues[i], point[i]);
        }
        for (int i = 0; i < point.length; ++i) {
            rangeSum += maxValues[i] - minValues[i];
        }
        return this;
    }

    @Override
    public BoundingBoxFloat addBox(AbstractBoundingBox<float[]> otherBox) {
        checkState(minValues != maxValues, "not a mutable box");
        rangeSum = 0;
        for (int i = 0; i < minValues.length; ++i) {
            minValues[i] = Math.min(minValues[i], otherBox.minValues[i]);
        }
        for (int i = 0; i < minValues.length; ++i) {
            maxValues[i] = Math.max(maxValues[i], otherBox.maxValues[i]);
        }
        for (int i = 0; i < minValues.length; ++i) {
            rangeSum += maxValues[i] - minValues[i];
        }
        return this;
    }

    @Override
    public int getDimensions() {
        return minValues.length;
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
    public boolean contains(float[] point) {
        checkArgument(point.length == minValues.length, " incorrect lengths");
        for (int i = 0; i < minValues.length; i++) {
            if (minValues[i] > point[i] || maxValues[i] < point[i]) {
                return false;
            }
        }

        return true;
    }

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
        if (!(other instanceof BoundingBoxFloat)) {
            return false;
        }

        AbstractBoundingBox<float[]> otherBox = (BoundingBoxFloat) other;
        return Arrays.equals(minValues, otherBox.minValues) && Arrays.equals(maxValues, otherBox.maxValues);
    }

}
