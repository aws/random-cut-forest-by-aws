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
import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

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
        checkArgument(capacity > 0, "capacity must be greater than 0");
        this.capacity = capacity;
        freeIndexes = new int[capacity];

        for (int j = 0; j < capacity; j++) {
            freeIndexes[j] = capacity - j - 1; // reverse order
        }

        freeIndexPointer = capacity - 1;
        occupied = new BitSet(capacity);
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
        /**
         * The below check will cause reInitialization to fail
         */
        // checkState(!occupied.get(index), "store tried to return an index marked
        // occupied");
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

    public List<Integer> getFreeIndices() {
        ArrayList result = new ArrayList<>();
        for (int i = 0; i <= freeIndexPointer; i++) {
            result.add(freeIndexes[i]);
        }
        return result;
    }

    public int[] getFreeIndexes() {
        return Arrays.copyOfRange(freeIndexes, 0, freeIndexPointer);
    }

    public void initializeIndices(List<Integer> freeList) {
        freeIndexPointer = freeList.size() - 1;
        for (int i = 0; i < freeList.size(); i++) {
            freeIndexes[i] = freeList.get(i);
            occupied.clear(i);
        }
    }
}
