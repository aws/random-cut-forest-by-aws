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

import java.util.Arrays;
import java.util.Stack;

/**
 * This class defines common functionality for Store classes, including
 * maintaining the stack of free pointers.
 */
// to be renamed to IndexIntervalManager in next PR alongside ser/de changes
public class IntervalManager {

    protected int capacity;
    protected int[] freeIndexesStart;
    protected int[] freeIndexesEnd;
    protected int lastInUse;

    public IntervalManager(int capacity) {
        checkArgument(capacity > 0, "incorrect parameters");
        freeIndexesEnd = new int[1];
        freeIndexesStart = new int[1];
        lastInUse = 1;
        this.capacity = capacity;
        freeIndexesStart[0] = 0;
        freeIndexesEnd[0] = capacity - 1;
    }

    public IntervalManager(Stack<int[]> stack, int capacity) {
        checkArgument(capacity > 0, "incorrect parameters");
        lastInUse = stack.size();
        freeIndexesEnd = new int[lastInUse + 1];
        freeIndexesStart = new int[lastInUse + 1];
        this.capacity = capacity;
        int count = 0;
        while (stack.size() > 0) {
            int[] interval = stack.pop();
            freeIndexesStart[count] = interval[0];
            freeIndexesEnd[count] = interval[1];
            ++count;
        }
    }

    public void extendCapacity(int newCapacity) {
        checkArgument(newCapacity > capacity, " incorrect call, we can only increase capacity");
        // the current capacity need not be the final capacity, for example in case of
        // point store
        if (freeIndexesStart.length == lastInUse) {
            freeIndexesStart = Arrays.copyOf(freeIndexesStart, lastInUse + 1);
            freeIndexesEnd = Arrays.copyOf(freeIndexesEnd, lastInUse + 1);
        }
        freeIndexesStart[lastInUse] = capacity;
        freeIndexesEnd[lastInUse] = (newCapacity - 1);
        lastInUse += 1;
        capacity = newCapacity;

    }

    public boolean isEmpty() {
        return (lastInUse == 0);
    }

    /**
     * @return the maximum number of nodes whose data can be stored.
     */
    public int getCapacity() {
        if (capacity == 0) {
            System.out.println("HUH");
        }
        return capacity;
    }

    /**
     * @return the number of nodes whose data is currently stored.
     */
    public int size() {
        return capacity - lastInUse;
    }

    /**
     * Take an index from the free index stack.
     * 
     * @return a free index that can be used to store a value.
     */
    public int takeIndex() {
        checkState(lastInUse > 0, "store is full");
        int answer = freeIndexesStart[lastInUse - 1];
        if (answer == freeIndexesEnd[lastInUse - 1]) {
            lastInUse -= 1;
        } else {
            freeIndexesStart[lastInUse - 1] = answer + 1;
        }
        return answer;
    }

    /**
     * Release an index. After the release, the index value may be returned in a
     * future call to {@link #takeIndex()}.
     * 
     * @param index The index value to release.
     */
    public void releaseIndex(int index) {
        if (lastInUse > 0) {
            int start = freeIndexesStart[lastInUse - 1];
            int end = freeIndexesEnd[lastInUse - 1];
            if (start == index + 1) {
                freeIndexesStart[lastInUse - 1] = index;
                return;
            } else if (end + 1 == index) {
                freeIndexesEnd[lastInUse - 1] = index;
                return;
            }
        }
        if (freeIndexesStart.length == lastInUse) {
            freeIndexesStart = Arrays.copyOf(freeIndexesStart, lastInUse + 1);
            freeIndexesEnd = Arrays.copyOf(freeIndexesEnd, lastInUse + 1);
        }

        freeIndexesStart[lastInUse] = index;
        freeIndexesEnd[lastInUse] = index;
        lastInUse += 1;
    }

}
