/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.sampler.WeightedPoint;
import com.amazon.randomcutforest.tree.RandomCutTree;

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

/**
 * TreeUpdater is a utility class to facilitate updating a {@link SimpleStreamSampler} and a {@link RandomCutTree} in a
 * single step.
 */
public class TreeUpdater {

    private final SimpleStreamSampler sampler;
    private final RandomCutTree tree;

    /**
     * Create a new TreeUpdater which links the given sampler and tree. In this TreeUpdater's {@link #update} method,
     * a point is first submitted to the sampler. If the point is accepted by the sampler, then the point is then
     * used to update the tree.
     *
     * @param sampler A stream sampler.
     * @param tree    A Random Cut Tree.
     */
    public TreeUpdater(SimpleStreamSampler sampler, RandomCutTree tree) {
        checkNotNull(sampler, "sampler must not be null");
        checkNotNull(tree, "tree must not be null");
        this.sampler = sampler;
        this.tree = tree;
    }

    /**
     * @return the sampler in this TreeUpdater.
     */
    public SimpleStreamSampler getSampler() {
        return sampler;
    }

    /**
     * @return the Random Cut Tree in this TreeUpdater.
     */
    public RandomCutTree getTree() {
        return tree;
    }

    /**
     * Update the TreeUpdater with the given point. The point is submitted to the sampler, and if it's accepted then
     * the sampler's evicted point is removed from the tree and the new point is added.
     *
     * @param point         The point being used to updated the sampler and tree.
     * @param sequenceIndex The ordinal when this point was added to the forest.
     */
    public void update(double[] point, long sequenceIndex) {
        /*
         * this execution should be sequential
         * at the current moment the sequence indexes are not used by the trees
         * but the trees can use both the sequence index and the sampling weight
         * as parts of internal logic
         */
        WeightedPoint candidate = sampler.sample(point, sequenceIndex);
        if (candidate != null) {
            WeightedPoint evictedPoint = sampler.getEvictedPoint();
            if (evictedPoint != null) {
                tree.deletePoint(evictedPoint);
            }
            tree.addPoint(candidate);
        }
    }
}
