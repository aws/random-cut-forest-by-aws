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

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

/**
 * This class defines common functionality for Store classes, including
 * maintaining the stack of free pointers.
 */
public class IndexManager {

    protected final int capacity;
    protected int[] freeIndexes;
    protected int freeIndexPointer;
    protected final BitSet occupied;

    /**
     * Create a new indexmanager with the given capacity.
     * 
     * @param capacity The total number of values that can be saved in this store.
     */
    public IndexManager(int capacity) {
        this(capacity, new BitSet(capacity));
    }

    /**
     * creates a new index manager with a given capacity and a bitset representing
     * the indices already in use
     * 
     * @param capacity total number of indices
     * @param bits     bitset correspond to the used indices, and null corresponding
     *                 to
     */
    public IndexManager(int capacity, BitSet bits) {
        checkArgument(capacity > 0, "capacity must be greater than 0");
        this.capacity = capacity;
        occupied = bits;
        freeIndexPointer = capacity - occupied.cardinality() - 1;
        if (freeIndexPointer != capacity - 1) {
            freeIndexes = new int[freeIndexPointer + 1];
            int location = 0;
            for (int i = capacity - 1; i >= 0; i--) {
                if (!occupied.get(i)) {
                    freeIndexes[location++] = i;
                }
            }
        } else {
            freeIndexes = new int[0];
        }

    }

    /**
     * Construct a new IndexManager with the given capacity, array of free indexes
     * and a value corresponding to the number of free indices
     *
     * @param capacity         the size of the index
     * @param freeIndexes      An array of unoccupied index values.
     * @param freeIndexPointer Entries in freeIndexes between 0 (inclusive) and
     *                         freeIndexPointer (inclusive) contain valid index
     *                         values. if freeIndexPointer is larger than the length
     *                         of the freeIndices array then the implcit guarantee
     *                         is that location i greater or equal
     *                         freeIndexes.length contains index value (capacity - i
     *                         -1)
     */
    public IndexManager(int capacity, int[] freeIndexes, int freeIndexPointer) {
        checkNotNull(freeIndexes, "freeIndexes must not be null");

        this.capacity = capacity;
        this.freeIndexes = freeIndexes;
        this.freeIndexPointer = freeIndexPointer;

        occupied = new BitSet(capacity);
        occupied.set(0, capacity);

        for (int i = 0; i < freeIndexPointer; i++) {
            if (i < freeIndexes.length) {
                occupied.clear(freeIndexes[i]);
            } else {
                occupied.clear(capacity - i - 1);
            }
        }

    }

    // the following is only used in testing
    public IndexManager(int[] freeIndexes, int freeIndexPointer) {
        this(freeIndexes.length, freeIndexes, freeIndexPointer);
    }

    public IndexManager(IndexManager manager, int newCapacity) {
        this(newCapacity);
        checkArgument(manager.occupied.cardinality() == manager.capacity, " incorrect application, not full");
        occupied.or(manager.occupied);
        freeIndexPointer = newCapacity - manager.capacity - 1;
    }

    public boolean isFull() {
        return (freeIndexPointer == -1);
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
        int index;
        if (freeIndexPointer < freeIndexes.length) {
            index = freeIndexes[freeIndexPointer--];
        } else {
            index = capacity - freeIndexPointer - 1;
            freeIndexPointer--;
        }
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
        if (freeIndexPointer + 1 >= freeIndexes.length) {
            if (index == capacity - (freeIndexPointer + 1) - 1) {
                ++freeIndexPointer;
                return;
            } else {
                int cap = Math.min(capacity, freeIndexPointer + 10);
                int oldLength = freeIndexes.length;
                freeIndexes = Arrays.copyOf(freeIndexes, cap);
                for (int j = oldLength; j < cap && j < freeIndexPointer + 1; j++) {
                    freeIndexes[j] = capacity - j - 1;
                }
            }
        }
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
        if (freeIndexPointer + 1 < freeIndexes.length) {
            return Arrays.copyOf(freeIndexes, freeIndexPointer + 1);
        } else {
            return freeIndexes;
        }
    }

}
