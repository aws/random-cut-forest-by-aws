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
 * A store for points of precision type P, which can be double[] or float[]
 * which can be added to a store by the update coordinator and made accessible
 * to the trees in a read only manner.
 * 
 * @param <Point> precision type
 */
public interface IPointStore<Point> extends IPointStoreView<Point> {
    /**
     * Adds to the store; there may be a loss of precision if enableFloat is on in
     * the Forest level. But external interface of the forest is double[]
     *
     * Note that delete is automatic, that is when no trees are accessing the point
     * 
     * @param point       point to be added
     * @param sequenceNum sequence number of the point
     * @return index of the stored point
     */
    int add(double[] point, long sequenceNum);

    // increments and returns the incremented value
    int incrementRefCount(int index);

    // decrements and returns the decremented value
    int decrementRefCount(int index);

}
