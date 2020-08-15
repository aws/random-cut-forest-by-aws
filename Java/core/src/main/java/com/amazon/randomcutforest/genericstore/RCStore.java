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

import static com.amazon.randomcutforest.CommonUtils.checkState;

/**
 * Reference counted store.
 * 
 * @param <T>
 */
public class RCStore<T> extends Store<T> {

    private final int[] refCount;

    public RCStore(int capacity) {
        super(capacity);
        refCount = new int[capacity];
    }

    @Override
    public int add(T t) {
        int index = super.add(t);
        refCount[index] = 1;
        return index;
    }

    public void incrementRefCount(int index) {
        checkValidIndex(index);
        refCount[index]++;
    }

    public void decrementRefCount(int index) {
        checkValidIndex(index);
        refCount[index]--;

        if (refCount[index] == 0) {
            remove(index);
        }
    }

    public int getRefCount(int index) {
        return refCount[index];
    }

    @Override
    protected void checkValidIndex(int index) {
        super.checkValidIndex(index);
        checkState(refCount[index] > 0, "index is occupied but reference count is nonpositive");
    }
}
