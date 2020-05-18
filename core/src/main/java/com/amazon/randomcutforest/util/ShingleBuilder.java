/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.util;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

/**
 * A utility class for creating shingled points, which are also referred to as
 * shingles. A shingle consists of multiple points appended together. If
 * individual points have n dimensions, and we include k points in a shingle,
 * then the shingle will have size n * m.
 *
 * There are two strategies for shingling: sliding and cyclic. In a sliding
 * shingle, new points are appended to the end of the shingle, and old points
 * are removed from the front. For example, if we have a shingle size of 4 which
 * currently contains the points a, b, c, and d, then we can represent the
 * shingle as abcd. The following schematic shows how the shingle is updated as
 * we add new points e and f.
 * 
 * <pre>
 *     abcd => bcde
 *     bcde => cdef
 * </pre>
 *
 * With cycling shingling, when a new point is added to a shingle it overwrites
 * the oldest point in the shingle. Using the same setup as above, a cyclic
 * shingle would be updated as follows:
 * 
 * <pre>
 *     abcd => ebcd
 *     ebcd => efcd
 * </pre>
 */
public class ShingleBuilder {

    /**
     * Number of dimensions of each point in the shingle.
     */
    private final int dimensions;

    /**
     * Number of points in the shingle.
     */
    private final int shingleSize;

    /**
     * A buffer containing points recently added to the shingle.
     */
    private final double[][] recentPoints;

    /**
     * A flag indicating whether we should use a cyclic shift or a linear shift when
     * creating shingles.
     */
    private final boolean cyclic;

    /**
     * The index where the next point will be copied to. This is equal to the index
     * of the oldest point currently in the shingle.
     */
    private int shingleIndex;

    /**
     * A flag indicating whether the shingle has been completely filled once.
     */
    private boolean full;

    /**
     * Create a new ShingleBuilder with the given dimensions and shingle size.
     * 
     * @param dimensions  The number of dimensions in the input points.
     * @param shingleSize The number of points to store in a shingle.
     * @param cyclic      If true, the shingle will use cyclic updates. If false, it
     *                    will use sliding updates.
     */
    public ShingleBuilder(int dimensions, int shingleSize, boolean cyclic) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        checkArgument(shingleSize > 0, "shingleSize must be greater than 0");

        this.dimensions = dimensions;
        this.shingleSize = shingleSize;
        this.cyclic = cyclic;
        recentPoints = new double[shingleSize][dimensions];

        shingleIndex = 0;
        full = false;
    }

    /**
     * Create a ShingleBuilder with the given dimensions and shingleSize. The
     * resulting builder uses sliding updates.
     * 
     * @param dimensions  The number of dimensions in the input points.
     * @param shingleSize The number of points to store in a shingle.
     */
    public ShingleBuilder(int dimensions, int shingleSize) {
        this(dimensions, shingleSize, false);
    }

    /**
     * @return true if the shingle has been completely filled once, false otherwise.
     */
    public boolean isFull() {
        return full;
    }

    /**
     * @return the number of dimensions in input points.
     */
    public int getInputPointSize() {
        return dimensions;
    }

    /**
     * @return the number of dimensions in a shingled point.
     */
    public int getShingledPointSize() {
        return dimensions * shingleSize;
    }

    /**
     * @return true if this ShingleBuilder uses cyclic updates, false otherwise.
     */
    public boolean isCyclic() {
        return cyclic;
    }

    /**
     * Return the index where the next input point will be stored in the internal
     * shingle buffer. If the ShingleBuilder uses cyclic updates, this value
     * indicates the current point in the cycle.
     *
     * @return the index where the next input point will be stored in the internal
     *         shingle buffer.
     */
    public int getShingleIndex() {
        return shingleIndex;
    }

    /**
     * Add a new point to this shingle. The point values are copied.
     * 
     * @param point The new point to be added to the shingle.
     */
    public void addPoint(double[] point) {
        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        System.arraycopy(point, 0, recentPoints[shingleIndex], 0, dimensions);

        shingleIndex = (shingleIndex + 1) % shingleSize;
        if (!full && shingleIndex == 0) {
            full = true;
        }
    }

    /**
     * @return the current shingled point.
     */
    public double[] getShingle() {
        double[] shingle = new double[shingleSize * dimensions];
        getShingle(shingle);
        return shingle;
    }

    /**
     * Write the current shingled point into the supplied buffer.
     * 
     * @param shingle A buffer where the shingled point will be written.
     */
    public void getShingle(double[] shingle) {
        checkNotNull(shingle, "shingle must not be null");
        checkArgument(shingle.length == dimensions * shingleSize, "shingle.length must be dimensions * shingleSize");

        int beginIndex = cyclic ? 0 : shingleIndex;

        for (int i = 0; i < shingleSize; i++) {
            System.arraycopy(recentPoints[(beginIndex + i) % shingleSize], 0, shingle, i * dimensions, dimensions);
        }
    }
}
