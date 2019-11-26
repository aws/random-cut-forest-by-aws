/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.returntypes;

import java.util.function.Function;

/**
 * An accumulator which can be used to short-circuit the number of trees visited if the responses from the trees seen
 * so far appear to be converging to a value.
 * for an example
 *
 * @param <R> The result type being accumulated.
 * @see com.amazon.randomcutforest.RandomCutForest#traverseForest(double[], Function, ConvergingAccumulator, Function)
 */
public interface ConvergingAccumulator<R> {
    /**
     * Add a new result value to this accumulator.
     *
     * @param value A single result value which should be accumulated together with other results.
     */
    void accept(R value);

    /**
     * @return 'true' if the accumulator has converged and we can stop accepting new values, 'false' otherwise.
     */
    boolean isConverged();

    /**
     * @return the number of values that have been accepted by this accumulator.
     */
    int getValuesAccepted();

    /**
     * @return the accumulated.
     */
    R getAccumulatedValue();
}
