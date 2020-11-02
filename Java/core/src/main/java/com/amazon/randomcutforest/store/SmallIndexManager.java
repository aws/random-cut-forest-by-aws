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
import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

/**
 * This class defines common functionality for Store classes, including
 * maintaining the stack of free pointers.
 */
public class SmallIndexManager {

    private final short capacity;
    protected final short[] freeIndexes;
    protected short freeIndexPointer;
    protected final BitSet occupied;

    /**
     * Create a new store with the given capacity.
     * 
     * @param capacity The total number of values that can be saved in this store.
     */
    public SmallIndexManager(short capacity) {
        this.capacity = capacity;
        freeIndexes = new short[capacity];

        for (int j = 0; j < capacity; j++) {
            freeIndexes[j] = (short) (capacity - j - 1); // reverse order
        }

        freeIndexPointer = (short) (capacity - 1);
        occupied = new BitSet(capacity);
    }

    /**
     * Construct a new SmallIndexManager with the given array of free indexes.
     * 
     * @param freeIndexes      An array of index values.
     * @param freeIndexPointer Entries in freeIndexes between 0 (inclusive) and
     *                         freeIndexPointer (inclusive) contain valid index
     *                         values.
     */
    public SmallIndexManager(short[] freeIndexes, short freeIndexPointer) {
        checkNotNull(freeIndexes, "freeIndexes must not be null");
        checkFreeIndexes(freeIndexes, freeIndexPointer);

        this.capacity = (short) freeIndexes.length;
        this.freeIndexes = freeIndexes;
        this.freeIndexPointer = freeIndexPointer;

        occupied = new BitSet(capacity);
        occupied.set(0, capacity);

        for (int i = 0; i <= freeIndexPointer; i++) {
            occupied.clear(freeIndexes[i]);
        }
    }

    private static void checkFreeIndexes(short[] freeIndexes, short freeIndexPointer) {
        checkArgument(0 <= freeIndexPointer && freeIndexPointer < freeIndexes.length,
                "freeIndexPoint must be between 0 (inclusive) and freeIndexes.length (exclusive)");

        int capacity = freeIndexes.length;
        Set<Short> freeIndexSet = new HashSet<>();

        for (int i = 0; i <= freeIndexPointer; i++) {
            short index = freeIndexes[i];
            checkArgument(!freeIndexSet.contains(index), "free index values must not be repeated");
            checkArgument(0 <= freeIndexes[i] && freeIndexes[i] < capacity,
                    "entries in freeIndexes must be between 0 (inclusive) and freeIndexes.length (exclusive)");
            freeIndexSet.add(index);
        }
    }

    /**
     * @return the maximum number of nodes whose data can be stored.
     */
    public short getCapacity() {
        return capacity;
    }

    /**
     * @return the number of nodes whose data is currently stored.
     */
    public short size() {
        return (short) (capacity - freeIndexPointer - 1);
    }

    /**
     * Take an index from the free index stack.
     * 
     * @return a free index that can be used to store a value.
     */
    protected short takeIndex() {
        checkState(freeIndexPointer >= 0, "store is full");
        short index = freeIndexes[freeIndexPointer--];
        checkState(!occupied.get(index), "store tried to return an index marked occupied");
        occupied.set(index);
        return index;
    }

    /**
     * Release an index. After the release, the index value may be returned in a
     * future call to {@link #takeIndex()}.
     * 
     * @param index The index value to release.
     */
    protected void releaseIndex(short index) {
        checkValidIndex(index);
        occupied.clear(index);
        freeIndexes[++freeIndexPointer] = index;
    }

    protected void checkValidIndex(int index) {
        checkArgument(index >= 0 && index < capacity, "index must be between 0 (inclusive) and capacity (exclusive)");
        checkArgument(occupied.get(index), "this index is not being used");
    }

    public short getFreeIndexPointer() {
        return freeIndexPointer;
    }

    public short[] getFreeIndexes() {
        return freeIndexes;
    }

}
