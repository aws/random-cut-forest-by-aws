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

package com.amazon.randomcutforest.anomalydetection;

import com.amazon.randomcutforest.returntypes.DiVector;

import java.util.function.BiFunction;

public class DynamicAttributionVisitor extends AbstractAttributionVisitor {

    /**
     * The function used to compute the base score in the case where the point being scored is equal to the leaf point
     * (provided the ignoreLeafEquals and ignoreLeafMassThreshold variables indicate that we should use this method).
     * <p>
     * Function arguments: leaf depth, leaf mass
     */
    private final BiFunction<Double, Double, Double> scoreSeen;

    /**
     * A damping function used to dilute the impact of a point with a large number of duplicates on the base score.
     * <p>
     * Function arguments: leaf mass, tree mass
     */
    private final BiFunction<Double, Double, Double> damp;

    /**
     * The scoring function to use when the point being scored is not equal to the leaf point, or when the points are
     * equal but the ignoreLeafEquals or ignoreLeafMassThreshold variable indicates that we should use the
     * scoreUnseen method.
     * <p>
     * Function arguments: leaf depth, leaf mass
     */
    private final BiFunction<Double, Double, Double> scoreUnseen;

    /**
     *
     * @param point to be scored
     * @param treeMass mass of the tree
     * @param ignoreLeafMassThreshold threshold of mass for leaves to be ignored
     * @param scoreSeen part of the score when point has been seen
     * @param scoreUnseen part of the score for unseen point
     * @param damp dampening function for seen points
     */
    public DynamicAttributionVisitor(double[] point, int treeMass,
            int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> scoreSeen,
            BiFunction<Double, Double, Double> scoreUnseen, BiFunction<Double, Double, Double> damp) {
        super(point, treeMass, ignoreLeafMassThreshold);
        this.scoreSeen = scoreSeen;
        this.scoreUnseen = scoreUnseen;
        this.damp = damp;
    }

    /**
     * Same as above with a default non-dampening
     * @param point to be scored
     * @param treeMass mass of the tree
     * @param ignoreLeafMassThreshold mass of the leaves to be ignored
     * @param scoreSeen score when point has been seen
     * @param scoreUnseen score when point has not been seen
      */
    public DynamicAttributionVisitor(double[] point, int treeMass,
                                     int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> scoreSeen,
                                     BiFunction<Double, Double, Double> scoreUnseen) {
        super(point, treeMass, ignoreLeafMassThreshold);
        this.scoreSeen = scoreSeen;
        this.scoreUnseen = scoreUnseen;
        this.damp = (x, y) -> 1.0;
    }



    @Override
    protected double scoreSeen(int depth, int leafMass) {
        return scoreSeen.apply((double) depth, (double) leafMass);
    }

    @Override
    protected double scoreUnseen(int depth, int leafMass) {
        return scoreUnseen.apply((double) depth, (double) leafMass);
    }

    @Override
    protected double damp(int leafMass, int treeMass) {
        return damp.apply((double) leafMass, (double) treeMass);
    }

    // turning off normalization

    @Override
    public DiVector getResult() {
        return new DiVector(directionalAttribution);
    }
}
