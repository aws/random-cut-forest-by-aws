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

/**
 * A Cut represents a division of space into two half-spaces. Cuts are used to
 * define the tree structure in {@link RandomCutTree}, and they determine the
 * standard tree traversal path defined in {@link RandomCutTree#traverse}.
 */
public class Cut {

    private final int dimension;
    private final double value;

    /**
     * Create a new Cut with the given dimension and value.
     *
     * @param dimension The 0-based index of the dimension that the cut is made in.
     * @param value     The spatial value of the cut.
     */
    public Cut(int dimension, double value) {
        this.dimension = dimension;
        this.value = value;
    }

    /**
     * For the given point, this method compares the value of that point in the cut
     * dimension to the cut value. If the point's value in the cut dimension is less
     * than or equal to the cut value this method returns true, otherwise it returns
     * false. The name of this method is a mnemonic: if we are working in a
     * one-dimensional space, then this method will return 'true' if the point value
     * is to the left of the cut value on the standard number line.
     *
     * @param point A point that we are testing in relation to the cut
     * @param cut   A Cut instance.
     * @return true if the value of the point coordinate corresponding to the cut
     *         dimension is less than or equal to the cut value, false otherwise.
     */
    public static boolean isLeftOf(double[] point, Cut cut) {
        return point[cut.getDimension()] <= cut.getValue();
    }

    /**
     * Return the index of the dimension that this cut was made in.
     *
     * @return the 0-based index of the dimension that this cut was made in.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Return the value of the cut. This value separates space into two half-spaces:
     * the set of points whose coordinate in the cut dimension is less than the cut
     * value, and the set of points whose coordinate in the cut dimension is greater
     * than the cut value.
     *
     * @return the value of the cut.
     */
    public double getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.format("Cut(%d, %f)", dimension, value);
    }
}
