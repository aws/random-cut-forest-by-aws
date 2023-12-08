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

package com.amazon.randomcutforest.returntypes;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collector;

import com.amazon.randomcutforest.tree.BoundingBox;

public class ConditionalTreeSample {

    /**
     * the index of the point in the PointStore which is used to construct the
     * sample for a query
     */
    public int pointStoreIndex;

    /**
     * the bounding box in the tree of the node which is the parent of the point
     * used to construct the sample Note that the bounding box is in the projective
     * space defined by the tree
     */
    protected BoundingBox parentOfLeafBox;

    /**
     * L1 distance of the sampled point (in the projective space of the tree) L1
     * distancce is chosen since the entire notion of RCF is oriented towards L1
     * sampling
     */

    public double distance;

    /**
     * the point in the tree corresponding to the sample
     */

    public float[] leafPoint;

    /**
     * weight of the point ; useful for deduplication -- this can also be resued if
     * trees are assigned weights
     */
    public double weight;

    public ConditionalTreeSample(int pointStoreIndex, BoundingBox box, double distance, float[] leafPoint) {
        this.pointStoreIndex = pointStoreIndex;
        this.parentOfLeafBox = box;
        this.distance = distance;
        this.leafPoint = leafPoint;
        this.weight = 1.0;
    }

    public static Collector<ConditionalTreeSample, ArrayList<ConditionalTreeSample>, ArrayList<ConditionalTreeSample>> collector = Collector
            .of(ArrayList::new, ArrayList::add, (left, right) -> {
                left.addAll(right);
                return left;
            }, list -> list);
    // the collector specifically does not try to sort/dedup since we could (and
    // would) be running the
    // collector in a parallel mode

    public static List<ConditionalTreeSample> dedup(List<ConditionalTreeSample> list) {
        list.sort(Comparator.comparingInt(o -> o.pointStoreIndex));
        List<ConditionalTreeSample> newList = new ArrayList<>();
        newList.add(list.get(0));
        for (int j = 1; j < list.size(); j++) {
            if (list.get(j).pointStoreIndex == newList.get(newList.size() - 1).pointStoreIndex) {
                newList.get(newList.size() - 1).weight += list.get(j).weight;
            } else {
                newList.add(list.get(j));
            }
        }
        return newList;
    }

}
