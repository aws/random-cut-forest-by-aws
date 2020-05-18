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

package com.amazon.randomcutforest.anomalydetection;

import com.amazon.randomcutforest.CommonUtils;

/**
 * This visitor computes a scalar anomaly score for a specified point. The basic
 * score computation is defined by {@link AbstractScalarScoreVisitor}, and this
 * class overrides the scoring functions so that input points that are more
 * likely to separated from in-sample points by a random cut receive a higher
 * anomaly score.
 *
 * While this basic algorithm produces good results when all the points in the
 * sample are distinct, it can produce unexpected results when a significant
 * portion of the points in the sample are duplicates. Therefore this class
 * supports different optional features for modifying the score produced when
 * the point being scored is equal to the leaf node in the traversal.
 */
public class AnomalyScoreVisitor extends AbstractScalarScoreVisitor {

    /**
     * Construct a new ScalarScoreVisitor
     *
     * @param pointToScore The point whose anomaly score we are computing
     * @param treeMass     The total mass of the RandomCutTree that is scoring the
     *                     point
     */
    public AnomalyScoreVisitor(double[] pointToScore, int treeMass) {
        super(pointToScore, treeMass);
    }

    /**
     * Construct a new ScalarScoreVisitor
     *
     * @param pointToScore            The point whose anomaly score we are computing
     * @param treeMass                The total mass of the RandomCutTree that is
     *                                scoring the point
     * @param ignoreLeafMassThreshold Is the maximum mass of the leaf which can be
     *                                ignored
     */
    public AnomalyScoreVisitor(double[] pointToScore, int treeMass, int ignoreLeafMassThreshold) {
        super(pointToScore, treeMass, ignoreLeafMassThreshold);
    }

    @Override
    protected double scoreSeen(int depth, int mass) {
        return CommonUtils.defaultScoreSeenFunction(depth, mass);
    }

    @Override
    protected double scoreUnseen(int depth, int mass) {
        return CommonUtils.defaultScoreUnseenFunction(depth, mass);
    }

    @Override
    protected double damp(int leafMass, int treeMass) {
        return CommonUtils.defaultDampFunction(leafMass, treeMass);
    }
}
