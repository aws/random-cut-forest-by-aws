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

/**
 * A read-only point store interface.
 */
public interface IPointStoreReadOnly<P> {
    int getDimensions();

    int getCapacity();

    boolean pointEquals(int index, P point);

    P get(int index);

    void incrementRefCount(int index);

    void decrementRefCount(int index);

    /**
     * Prints the point given the index, irrespective of the encoding of the point.
     * Used in exceptions and error messages
     * 
     * @param index index of the point in the store
     * @return a string that can be printed
     */
    String toString(int index);

}
