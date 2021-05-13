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


package com.amazon.randomcutforest;


import com.amazon.randomcutforest.tree.IBoundingBoxView;
import java.util.Objects;

/** A collection of common utility functions. */
public class CommonUtils {

    private CommonUtils() {}

    /**
     * Throws an {@link IllegalArgumentException} with the specified message if the specified input
     * is false.
     *
     * @param condition A condition to test.
     * @param message The error message to include in the {@code IllegalArgumentException} if {@code
     *     condition} is false.
     * @throws IllegalArgumentException if {@code condition} is false.
     */
    public static void checkArgument(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }

    /**
     * Throws an {@link IllegalStateException} with the specified message if the specified input is
     * false.
     *
     * @param condition A condition to test.
     * @param message The error message to include in the {@code IllegalStateException} if {@code
     *     condition} is false.
     * @throws IllegalStateException if {@code condition} is false.
     */
    public static void checkState(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    /**
     * Throws an {@link IllegalStateException} with the specified message if the specified input is
     * false. This would eventually become asserts.
     *
     * @param condition A condition to test.
     * @param message The error message to include in the {@code IllegalStateException} if {@code
     *     condition} is false.
     * @throws IllegalStateException if {@code condition} is false.
     */
    public static void validateInternalState(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    /**
     * Throws a {@link NullPointerException} with the specified message if the specified input is
     * null.
     *
     * @param <T> An arbitrary type.
     * @param object An object reference to test for nullity.
     * @param message The error message to include in the {@code NullPointerException} if {@code
     *     object} is null.
     * @return {@code object} if not null.
     * @throws NullPointerException if the supplied object is null.
     */
    public static <T> T checkNotNull(T object, String message) {
        Objects.requireNonNull(object, message);
        return object;
    }

    /**
     * Compute the probability of separation for a bounding box adn a point. This method considers
     * the bounding box created by merging the query point into the existing bounding box, and
     * computes the probability that a random cut would separate the query point from the merged
     * bounding box.
     *
     * @param boundingBox is the bounding box used in RandomCutTree
     * @param queryPoint is the multidimensional point
     * @return the probability of separation choosing a random cut
     */
    public static double getProbabilityOfSeparation(
            final IBoundingBoxView boundingBox, double[] queryPoint) {
        double sumOfNewRange = 0d;
        double sumOfDifferenceInRange = 0d;

        for (int i = 0; i < queryPoint.length; ++i) {
            double maxVal = boundingBox.getMaxValue(i);
            double minVal = boundingBox.getMinValue(i);
            double oldRange = maxVal - minVal;

            if (maxVal < queryPoint[i]) {
                maxVal = queryPoint[i];
            } else if (minVal > queryPoint[i]) {
                minVal = queryPoint[i];
            } else {
                sumOfNewRange += oldRange;
                continue;
            }

            double newRange = maxVal - minVal;
            sumOfNewRange += newRange;
            sumOfDifferenceInRange += (newRange - oldRange);
        }

        if (sumOfNewRange <= 0) {
            return 0;
        } else return sumOfDifferenceInRange / sumOfNewRange;
    }

    /**
     * The default anomaly scoring function for points that contained in a tree.
     *
     * @param depth The depth of the leaf node where this method is invoked
     * @param mass The number of times the point has been seen before
     * @return The score contribution from this previously-seen point
     */
    public static double defaultScoreSeenFunction(double depth, double mass) {
        return 1.0 / (depth + Math.log(mass + 1.0) / Math.log(2.0));
    }

    /**
     * The default anomaly scoring function for points not already contained in a tree.
     *
     * @param depth The depth of the leaf node where this method is invoked
     * @param mass The number of times the point has been seen before
     * @return The score contribution from this point
     */
    public static double defaultScoreUnseenFunction(double depth, double mass) {
        return 1.0 / (depth + 1);
    }

    public static double defaultDampFunction(double leafMass, double treeMass) {
        return 1.0 - leafMass / (2 * treeMass);
    }

    /**
     * Some algorithms which return a scalar value need to scale that value by tree mass for
     * consistency. This is the default method for computing the scale factor in these cases. The
     * function has to be associative in its first argument (when the second is fixed) That is, fn
     * (x1, y) + fn (x2, y) = fn (x1 + x2, y)
     *
     * @param scalarValue The value being scaled
     * @param mass The mass of the tree where this method is invoked
     * @return The original value scaled appropriately for this tree
     */
    public static double defaultScalarNormalizerFunction(double scalarValue, double mass) {
        return scalarValue * Math.log(mass + 1) / Math.log(2.0);
    }

    /**
     * The following function forms the core of RCFs, given a BoundingBox it produces the
     * probability of cutting in different dimensions. While this function is absorbed in the logic
     * of the different simpler scoring methods, the scoring methods that are mode advanced (for
     * example, trying to simulate an Transductive Isolation Forest with streaming) require this
     * function. A different function can be used to simulate via non-RCFs.
     *
     * @param boundingBox bounding box of a set of points
     * @return array of probabilities of cutting in that specific dimension
     */
    public static double[] defaultRCFgVecFunction(IBoundingBoxView boundingBox) {
        double[] answer = new double[boundingBox.getDimensions()];

        for (int i = 0; i < boundingBox.getDimensions(); ++i) {
            double maxVal = boundingBox.getMaxValue(i);
            double minVal = boundingBox.getMinValue(i);
            double oldRange = maxVal - minVal;

            if (oldRange > 0) {
                answer[i] = oldRange;
            }
        }
        return answer;
    }
    ;

    public static double[] toDoubleArray(float[] point) {
        checkNotNull(point, "point must not be null");
        double[] result = new double[point.length];
        for (int i = 0; i < point.length; i++) {
            result[i] = point[i];
        }
        return result;
    }

    public static float[] toFloatArray(double[] point) {
        checkNotNull(point, "point must not be null");
        float[] result = new float[point.length];
        for (int i = 0; i < point.length; i++) {
            result[i] = (float) point[i];
        }
        return result;
    }
}
