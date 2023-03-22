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

package com.amazon.randomcutforest.tree;

import com.amazon.randomcutforest.config.IDynamicConfig;
import com.amazon.randomcutforest.executor.ITraversable;

/**
 * A tree that can potentially interact with a coordinator
 *
 * @param <PointReference> The internal point representation expected by the
 *                         component models in this list.
 * @param <Point>          The explicit data type of points being passed
 */
public interface ITree<PointReference, Point> extends ITraversable, IDynamicConfig {
    int getMass();

    float[] projectToTree(float[] point);

    float[] liftFromTree(float[] result);

    double[] liftFromTree(double[] result);

    int[] projectMissingIndices(int[] list);

    PointReference addPoint(PointReference point, long sequenceIndex);

    void addPointToPartialTree(PointReference point, long sequenceIndex);

    void validateAndReconstruct();

    PointReference deletePoint(PointReference point, long sequenceIndex);

    default long getRandomSeed() {
        return 0L;
    }
}
