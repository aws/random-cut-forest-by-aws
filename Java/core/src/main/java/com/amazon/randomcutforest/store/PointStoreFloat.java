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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

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
public class PointStoreFloat extends PointStore<float[]> implements IPointStore<float[]> {

    protected final float[] store;

    /**
     * Create a new PointStore with the given dimensions and capacity.
     *
     * @param dimensions The number of dimensions in stored points.
     * @param capacity   The maximum number of points that can be stored.
     */
    public PointStoreFloat(int dimensions, int capacity) {
        super(dimensions, capacity);
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        store = new float[capacity * dimensions];
    }

    public PointStoreFloat(float[] store, short[] refCount, int[] freeIndexes, int freeIndexPointer) {
        super(store.length / refCount.length, refCount, freeIndexes, freeIndexPointer);
        checkNotNull(store, "store must not be null");
        checkNotNull(refCount, "refCount must not be null");
        checkArgument(refCount.length == capacity, "refCount.length must equal capacity");
        checkArgument(store.length % capacity == 0, "store.length must be an exact multiple of capacity");

        this.store = store;
    }

    /**
     * Add a point to the point store and return the index of the stored point.
     *
     * @param point The point being added to the store.
     * @return the index value of the stored point.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     * @throws IllegalStateException    if the point store is full.
     */
    public int add(double[] point) {
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");

        int nextIndex = takeIndex();
        for (int i = 0; i < dimensions; i++) {
            store[nextIndex * dimensions + i] = (float) point[i];
        }

        refCount[nextIndex] = 1;
        return nextIndex;
    }

    /**
     * Test whether the given point is equal to the point stored at the given index.
     * This operation uses point-wise <code>==</code> to test for equality.
     *
     * @param index The index value of the point we are comparing to.
     * @param point The point we are comparing for equality.
     * @return true if the point stored at the index is equal to the given point,
     *         false otherwise.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non-positive.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     */

    @Override
    public boolean pointEquals(int index, float[] point) {
        checkValidIndex(index);
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");

        for (int j = 0; j < dimensions; j++) {
            if (point[j] != store[j + index * dimensions]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Get a copy of the point at the given index.
     *
     * @param index An index value corresponding to a storage location in this point
     *              store.
     * @return a copy of the point stored at the given index.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non-positive.
     */
    @Override
    public float[] get(int index) {
        checkValidIndex(index);
        return Arrays.copyOfRange(store, index * dimensions, (index + 1) * dimensions);
    }

    /**
     * print the point at location index
     * 
     * @param index index of the point in the store
     * @return ascii output
     */
    @Override
    public String toString(int index) {
        return Arrays.toString(get(index));
    }

    public float[] getStore() {
        return store;
    }

}
