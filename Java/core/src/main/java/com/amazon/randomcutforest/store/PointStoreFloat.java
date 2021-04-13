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

import java.util.Arrays;

/**
 * PointStoreFloat is a PointStore defined on base type FLoat
 */
public class PointStoreFloat extends PointStore<float[], float[]> {

    public PointStoreFloat(Builder builder) {
        super(builder);
        checkArgument(builder.store == null || builder.store.length == currentStoreCapacity * dimensions,
                " incorrect store length");
        store = (builder.store == null) ? new float[currentStoreCapacity * dimensions] : builder.store;
    }

    public PointStoreFloat(int dimensions, int capacity) {
        this(new Builder().dimensions(dimensions).shingleSize(1).capacity(capacity).indexCapacity(capacity)
                .currentStoreCapacity(capacity));
    }

    @Override
    void resizeStore() {
        int maxCapacity = (rotationEnabled) ? 2 * capacity : capacity;
        int newCapacity = Math.min(2 * currentStoreCapacity, maxCapacity);
        if (newCapacity > currentStoreCapacity) {
            float[] newStore = new float[newCapacity * dimensions];
            System.arraycopy(store, 0, newStore, 0, currentStoreCapacity * dimensions);
            currentStoreCapacity = newCapacity;
            store = newStore;
        }
    }

    @Override
    boolean checkShingleAlignment(int location, double[] point) {
        boolean test = (location - dimensions + baseDimension >= 0);
        for (int i = 0; i < dimensions - baseDimension && test; i++) {
            test = (((float) point[i]) == store[location - dimensions + baseDimension + i]);
        }
        return test;
    }

    @Override
    void copyPoint(double[] point, int src, int location, int length) {
        for (int i = 0; i < length; i++) {
            store[location + i] = (float) point[src + i];
        }
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
     *                                  index is nonpositive.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     */

    @Override
    public boolean pointEquals(int index, float[] point) {
        indexManager.checkValidIndex(index);
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");
        int address = directLocationMap ? index * dimensions : locationList[index];
        if (!rotationEnabled) {
            for (int j = 0; j < dimensions; j++) {
                if (point[j] != store[j + address]) {
                    return false;
                }
            }
        } else {
            for (int j = 0; j < dimensions; j++) {
                if (point[j] != store[(j + address) % dimensions]) {
                    return false;
                }
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
     *                                  index is nonpositive.
     */
    @Override
    public float[] get(int index) {
        indexManager.checkValidIndex(index);
        int address = locationList[index];
        if (!rotationEnabled) {
            return Arrays.copyOfRange(store, address, address + dimensions);
        } else {
            float[] answer = new float[dimensions];
            for (int i = 0; i < dimensions; i++) {
                answer[(address + i) % dimensions] = store[address + i];
            }
            return answer;
        }
    }

    public float[] getScaledPoint(int index, double factor) {
        float[] answer = get(index);
        for (int i = 0; i < dimensions; i++) {
            answer[i] *= factor;
        }
        return answer;
    }

    @Override
    public String toString(int index) {
        return Arrays.toString(get(index));
    }

    @Override
    void copyTo(int dest, int source, int length) {
        checkArgument(dest <= source, "error");
        for (int i = 0; i < length; i++) {
            store[dest + i] = store[source + i];
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends PointStore.Builder<Builder> {
        private float[] store = null;

        public PointStoreFloat.Builder store(float[] store) {
            this.store = store;
            return this;
        }

        public PointStoreFloat build() {
            return new PointStoreFloat(this);
        }
    }
}
