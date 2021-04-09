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
public class IndexManager {

    protected final int capacity;
    protected final int[] freeIndexes;
    protected int freeIndexPointer;
    protected final BitSet occupied;

    /**
     * Create a new store with the given capacity.
     * 
     * @param capacity The total number of values that can be saved in this store.
     */
    public IndexManager(int capacity) {
        this(capacity, new BitSet(capacity));
    }

    public IndexManager(int capacity, BitSet occupied) {
        checkArgument(capacity > 0, "capacity must be greater than 0");
        this.capacity = capacity;
        freeIndexes = new int[capacity];
        this.occupied = occupied;

        for (int j = 0; j < capacity; j++) {
            freeIndexes[j] = capacity - j - 1; // reverse order
        }
        freeIndexPointer = capacity - 1;

    }

    /**
     * Construct a new IndexManager with the given array of free indexes.
     * 
     * @param freeIndexes      An array of index values.
     * @param freeIndexPointer Entries in freeIndexes between 0 (inclusive) and
     *                         freeIndexPointer (inclusive) contain valid index
     *                         values.
     */

    public IndexManager(int[] freeIndexes, int freeIndexPointer) {
        checkNotNull(freeIndexes, "freeIndexes must not be null");
        checkFreeIndexes(freeIndexes, freeIndexPointer);

        this.capacity = freeIndexes.length;
        this.freeIndexes = freeIndexes;
        this.freeIndexPointer = freeIndexPointer;

        occupied = new BitSet(capacity);
        occupied.set(0, capacity);

        for (int i = 0; i <= freeIndexPointer; i++) {
            occupied.clear(freeIndexes[i]);
        }
    }

    private static void checkFreeIndexes(int[] freeIndexes, int freeIndexPointer) {
        checkArgument(-1 <= freeIndexPointer && freeIndexPointer < freeIndexes.length,
                "freeIndexPointer must be between -1 (inclusive) and freeIndexes.length (exclusive)");

        int capacity = freeIndexes.length;
        Set<Integer> freeIndexSet = new HashSet<>();

        for (int i = 0; i <= freeIndexPointer; i++) {
            int index = freeIndexes[i];
            checkArgument(!freeIndexSet.contains(index), "free index values must not be repeated");
            checkArgument(0 <= freeIndexes[i] && freeIndexes[i] < capacity,
                    "entries in freeIndexes must be between 0 (inclusive) and freeIndexes.length (exclusive)");
            freeIndexSet.add(index);
        }
    }

    /**
     * @return the maximum number of nodes whose data can be stored.
     */
    public int getCapacity() {
        return capacity;
    }

    /**
     * @return the number of nodes whose data is currently stored.
     */
    public int size() {
        return capacity - freeIndexPointer - 1;
    }

    /**
     * Take an index from the free index stack.
     * 
     * @return a free index that can be used to store a value.
     */
    protected int takeIndex() {
        checkState(freeIndexPointer >= 0, "store is full");
        int index = freeIndexes[freeIndexPointer--];
        occupied.set(index);
        return index;
    }

    /**
     * Release an index. After the release, the index value may be returned in a
     * future call to {@link #takeIndex()}.
     * 
     * @param index The index value to release.
     */
    protected void releaseIndex(int index) {
        checkValidIndex(index);
        occupied.clear(index);
        freeIndexes[++freeIndexPointer] = index;
    }

    protected void checkValidIndex(int index) {
        checkArgument(index >= 0 && index < capacity, "index must be between 0 (inclusive) and capacity (exclusive)");
        checkArgument(occupied.get(index), "this index is not being used");
    }

    public int getFreeIndexPointer() {
        return freeIndexPointer;
    }

    public int[] getFreeIndexes() {
        return freeIndexes;
    }

}
