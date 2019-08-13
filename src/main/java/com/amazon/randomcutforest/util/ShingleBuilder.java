/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
     * A flag indicating whether we should use a cyclic shift or a linear shift when creating shingles.
     */
    private final boolean cyclic;

    /**
     * The index where the next point will be copied to. This is equal to the index of the oldest point currently in
     * the shingle.
     */
    private int shingleIndex;

    /**
     * A flag indicating whether the shingle has been completely filled once.
     */
    private boolean full;

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

    public ShingleBuilder(int dimensions, int shingleSize) {
        this(dimensions, shingleSize, false);
    }

    /**
     * Returns true if the shingle has been completely filled once, false otherwise.
     *
     * @return true if the shingle has been completely filled once, false otherwise.
     */
    public boolean isFull() {
        return full;
    }

    public int getInputPointSize() {
        return dimensions;
    }

    public int getShingledPointSize() {
        return dimensions * shingleSize;
    }

    public boolean isCyclic() {
        return cyclic;
    }

    public int getShingleIndex() {
        return shingleIndex;
    }

    public void addPoint(double[] point) {
        checkNotNull(point, "point must not be null");
        checkArgument(point.length == dimensions, String.format("point.length must equal %d", dimensions));
        System.arraycopy(point, 0, recentPoints[shingleIndex], 0, dimensions);

        shingleIndex = (shingleIndex + 1) % shingleSize;
        if (!full && shingleIndex == 0) {
            full = true;
        }
    }

    public double[] getShingle() {
        double[] shingle = new double[shingleSize * dimensions];
        getShingle(shingle);
        return shingle;
    }

    public void getShingle(double[] shingle) {
        checkNotNull(shingle, "shingle must not be null");
        checkArgument(shingle.length == dimensions * shingleSize, "shingle.length must be dimensions * shingleSize");

        int beginIndex = cyclic ? 0 : shingleIndex;

        for (int i = 0; i < shingleSize; i++) {
            System.arraycopy(recentPoints[(beginIndex + i) % shingleSize], 0, shingle, i * dimensions, dimensions);
        }
    }
}
