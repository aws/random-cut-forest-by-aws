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
public class SmallIndexManager {

    protected final short capacity;
    protected short[] freeIndexes;
    protected short freeIndexPointer;
    protected final BitSet occupied;

    public SmallIndexManager(short capacity) {
        this(capacity, null);
    }

    /**
     * this constructor sets up a SmallIndexManager based on a bitset that informs
     * which indices are already in use; the bitset is used internally as well.
     * 
     * @param capacity the maximum number of indices
     * @param bits     bitset indicating the indices already in use
     */
    public SmallIndexManager(short capacity, BitSet bits) {
        this.capacity = capacity;
        if (bits == null) {
            freeIndexes = new short[0];
            freeIndexPointer = (short) (capacity - 1);
            occupied = new BitSet(capacity);
        } else {
            /**
             * The stack may be implicitly defined. if freeIndexPointer exceeds
             * freeIndexes.length, then any intermediate location say i, must contain entry
             * capacity - i - 1
             */
            freeIndexPointer = -1;
            occupied = bits;
            for (int i = 0; i < capacity; i++) {
                if (!occupied.get(i)) {
                    freeIndexPointer++;
                }
            }
            if (freeIndexPointer != capacity - 1) {
                freeIndexes = new short[freeIndexPointer + 1];
                int location = 0;
                for (int i = capacity - 1; i >= 0; i--) {
                    if (!occupied.get(i)) {
                        freeIndexes[location++] = (short) i;
                    }
                }
            } else {
                freeIndexes = new short[0];
            }
        }
    }

    /**
     * Construct a new SmallIndexManager with the given array of free indexes.
     * 
     * @param freeIndexes      An array of index values.
     * @param freeIndexPointer Entries in freeIndexes between 0 (inclusive) and
     *                         freeIndexPointer (inclusive) contain valid index
     *                         values.
     */

    public SmallIndexManager(int capacity, short[] freeIndexes, short freeIndexPointer) {
        checkNotNull(freeIndexes, "freeIndexes must not be null");

        this.capacity = (short) capacity;
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

    // as above, used for the existing tests
    public SmallIndexManager(short[] freeIndexes, short freeIndexPointer) {
        this(freeIndexes.length, freeIndexes, freeIndexPointer);
    }

    private static void checkFreeIndexes(short[] freeIndexes, short freeIndexPointer) {
        checkArgument(-1 <= freeIndexPointer && freeIndexPointer < freeIndexes.length,
                "freeIndexPointer must be between -1 (inclusive) and freeIndexes.length (exclusive)");

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
     * Take an index from the free index stack. Note that the stack can be implicit.
     * 
     * @return a free index that can be used to store a value.
     */
    protected short takeIndex() {
        checkState(freeIndexPointer >= 0, "store is full");
        int index;

        if (freeIndexPointer < freeIndexes.length) {
            index = freeIndexes[freeIndexPointer--];
        } else {
            index = capacity - freeIndexPointer - 1;
            freeIndexPointer--;
        }
        occupied.set(index);
        return (short) index;
    }

    /**
     * Release an index. After the release, the index value may be returned in a
     * future call to {@link #takeIndex()}.
     * 
     * @param index The index value to release.
     */
    protected void releaseIndex(int index) {
        checkArgument(index >= 0 && index < capacity, " incorrect index to release");
        checkArgument(occupied.get(index), " releasing the index twice");
        occupied.clear(index);
        if (freeIndexPointer + 1 >= freeIndexes.length) {
            if (index == capacity - (freeIndexPointer + 1) - 1) {
                ++freeIndexPointer;
                return;
            } else {
                // for most of the applications of this small sampler we would need one extra
                // entry; but freeIndexPointer can be -1 as well
                // this can be changed to a larger value to reduce repeated
                // allocation/deallocation
                // we are populating these intermediate values because the newly released
                // "index"
                // breaks the implicit guarantee maintained so far.
                int cap = Math.min(capacity, freeIndexPointer + 2);
                int oldLength = freeIndexes.length;
                freeIndexes = Arrays.copyOf(freeIndexes, cap);
                for (int j = oldLength; j < cap && j < freeIndexPointer + 1; j++) {
                    freeIndexes[j] = (short) (capacity - j - 1);
                }
            }
        }
        freeIndexes[++freeIndexPointer] = (short) index;
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
