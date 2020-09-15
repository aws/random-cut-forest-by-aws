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

package com.amazon.randomcutforest;

import com.amazon.randomcutforest.tree.SamplingTree;

public interface IUpdatable<P> {
    /**
     * Update the model with the given point. If a point is removed from the model
     * as part of the update, then it is returned. Otherwise the return value is
     * null.
     *
     * A given call to `update` may choose not to do anything with the point (and
     * therefore the model state may not change). For example, in a
     * {@link SamplingTree} the tree is only modified if the point is accepted by
     * the sampler.
     *
     * @param point The point submitted to the model
     * @return the point that was removed from the model as part of the update, or
     *         null if no point was removed.
     */
    P update(P point);
}
