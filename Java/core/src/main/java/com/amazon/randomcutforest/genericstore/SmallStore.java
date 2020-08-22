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

package com.amazon.randomcutforest.genericstore;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.lang.reflect.Array;
import java.util.BitSet;

/**
 * This class provides a fixed amount of storage slots for a given type. When an
 * object is stored, the class returns an index value that can be used to
 * retrieve the object later. The purpose of this class is to reduce the overall
 * memory footprint of an applicaiton by allowing classes to keep index values
 * instead of pointers. The index values used in this class are shorts, which
 * occupy 2 bytes (Java pointers occupy 8 bytes). Using shorts as index values,
 * this class has a maximum capacity of 32,767. If more capacity is needed, see
 * the {@link Store} class which uses 4 byte ints as index values.
 *
 * @param <T> The type of value being stored.
 */
public class SmallStore<T> {

    private final short capacity;
    private final T[] store;
    private final short[] freeBlockStack;
    private short freeBlockPointer;
    private final BitSet occupied;

    /**
     * Create a new small store with the given capacity.
     * 
     * @param capacity The maximum number of objects that can be added to this
     *                 store.
     */
    public SmallStore(Class<T> clazz, short capacity) {
        checkArgument(capacity > 0, "capacity must be greater than 0");
        this.capacity = capacity;

        @SuppressWarnings("unchecked")
        final T[] temp = (T[]) Array.newInstance(clazz, capacity);
        store = temp;

        freeBlockStack = new short[capacity];
        for (short j = 0; j < capacity; j++) {
            freeBlockStack[j] = (short) (capacity - j - 1); // reverse order
        }
        freeBlockPointer = (short) (capacity - 1);
        occupied = new BitSet(capacity);
    }

    /**
     * @return the maximum number of objects that can be stored.
     */
    public short getCapacity() {
        return capacity;
    }

    /**
     * @return the number of objects currently being stored.
     */
    public short size() {
        return (short) (capacity - freeBlockPointer - 1);
    }

    /**
     * Save a value to this store and return an index which can be used to look up
     * the value later.
     * 
     * @param t The value being stored.
     * @return an index that can be used to look up the value later.
     */
    public short add(T t) {
        checkState(freeBlockPointer >= 0, "store is full");
        short index = freeBlockStack[freeBlockPointer--];
        checkState(!occupied.get(index), "store tried to return an index marked occupied");
        occupied.set(index);
        store[index] = t;
        return index;
    }

    /**
     * Get the value at the given index.
     * 
     * @param index The index where the desired value is being stored.
     * @return the value at the given index.
     */
    public T get(short index) {
        checkValidIndex(index);
        return store[index];
    }

    /**
     * Release the value at the given index. Once released, this index may be used
     * to store another value.
     * 
     * @param index The index whose value we want to release.
     */
    public void remove(short index) {
        checkValidIndex(index);
        occupied.clear(index);
        freeBlockStack[++freeBlockPointer] = index;
    }

    /**
     * Check that the index falls within the correct bounds, and that there is
     * currently a value stored at the index.
     * 
     * @param index The index value to validate.
     * @throws IllegalArgumentException if the index value is less than 0 or greater
     *                                  than equal to capacity.
     * @throws IllegalArgumentException if no value is currently stored at the
     *                                  index.
     */
    protected void checkValidIndex(short index) {
        checkArgument(index >= 0 && index < capacity, "index must be between 0 (inclusive) and capacity (exclusive)");
        checkArgument(occupied.get(index), "no value is currently stored at this index");
    }
}
