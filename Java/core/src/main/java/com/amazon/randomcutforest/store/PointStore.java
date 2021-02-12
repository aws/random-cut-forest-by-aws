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

package com.amazon.randomcutforest.store;

import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.Arrays;

/**
 * PointStore is a fixed size repository of points, where each point is a float
 * array of a specified length. A PointStore counts references to points that
 * are added, and frees space internally when a given point is no longer in use.
 * The primary use of this store is to enable compression since the points in
 * two different trees do not have to be stored separately.
 *
 * Stored points are referenced by index values which can be used to look up the
 * point values and increment and decrement reference counts. Valid index values
 * are between 0 (inclusive) and capacity (exclusive).
 */
public abstract class PointStore<Point> extends IndexManager implements IPointStore<Point> {

    protected final short[] refCount;
    protected final int dimensions;

    /**
     * Create a new PointStore with the given dimensions and capacity.
     *
     * @param dimensions The number of dimensions in stored points.
     * @param capacity   The maximum number of points that can be stored.
     */
    public PointStore(int dimensions, int capacity) {
        super(capacity);
        this.dimensions = dimensions;
        refCount = new short[capacity];
    }

    public PointStore(int dimensions, short[] refCount, int[] freeIndexes, int freeIndexPointer) {
        super(freeIndexes, freeIndexPointer);
        this.dimensions = dimensions;
        this.refCount = refCount;
    }

    /**
     * @return the number of dimensions in stored points for this PointStore.
     */
    @Override
    public int getDimensions() {
        return dimensions;
    }

    /**
     * @param index The index value.
     * @return the reference count for the given index. The value 0 indicates that
     *         there is no point stored at that index.
     */
    public int getRefCount(int index) {
        return refCount[index];
    }

    /**
     * Increment the reference count for the given index. This operation assumes
     * that there is currently a point stored at the given index and will throw an
     * exception if that's not the case.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    public int incrementRefCount(int index) {
        checkValidIndex(index);
        return ++refCount[index];
    }

    /**
     * Decrement the reference count for the given index.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    public int decrementRefCount(int index) {
        checkValidIndex(index);

        if (refCount[index] == 1) {
            releaseIndex(index);
        }

        return --refCount[index];
    }

    /**
     * Test whether the given point is equal to the point stored at the given index.
     * This operation uses pointwise <code>==</code> to test for equality.
     *
     * @param index The index value of the point we are comparing to.
     * @param point The point we are comparing for equality.
     * @return true if the point stored at the index is equal to the given point,
     *         false otherwise.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     */

    // @Override
    // public abstract boolean pointEquals(int index, Point point);

    /**
     * Get a copy of the point at the given index.
     * 
     * @param index An index value corresponding to a storage location in this point
     *              store.
     * @return a copy of the point stored at the given index.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    @Override
    public abstract Point get(int index);

    @Override
    public abstract String toString(int index);

    @Override
    protected void checkValidIndex(int index) {
        super.checkValidIndex(index);
        checkState(refCount[index] > 0, "ref count at occupied index is 0");
    }

    public int getValidPrefix() {
        int prefix = capacity;
        while (prefix > 0 && !occupied.get(prefix - 1))
            prefix--;
        return prefix;
    }

    public short[] getRefCount() {
        return Arrays.copyOf(refCount, getValidPrefix());
    }
}
