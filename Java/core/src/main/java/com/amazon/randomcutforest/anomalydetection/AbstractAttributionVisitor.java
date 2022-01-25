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

package com.amazon.randomcutforest.anomalydetection;

import java.util.Arrays;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.tree.IBoundingBoxView;
import com.amazon.randomcutforest.tree.INodeView;

/**
 * Attribution exposes the attribution of scores produced by ScalarScoreVisitor
 * corresponding to different attributes. It allows a boolean
 * ignoreClosestCandidate; which when true will compute the attribution as it
 * that near neighbor was not present in RCF. This is turned on by default for
 * duplicate points seen by the forest, so that the attribution does not change
 * is a sequence of duplicate points are seen. For non-duplicate points, if the
 * boolean turned on, reduces effects of masking (when anomalous points are
 * included in the forest (which will be true with a few samples or when the
 * samples are not refreshed appropriately). It is worth remembering that
 * disallowing anomalous points from being included in the forest forest
 * explicitly will render the algorithm incapable of adjusting to a new normal
 * -- which is a strength of this algorithm.
 **/
public abstract class AbstractAttributionVisitor implements Visitor<DiVector> {

    public static final int DEFAULT_IGNORE_LEAF_MASS_THRESHOLD = 0;

    protected final double[] differenceInRangeVector;
    protected final float[] pointToScore;
    protected final int treeMass;
    protected final DiVector directionalAttribution;
    protected boolean hitDuplicates;
    protected double savedScore;
    protected double sumOfNewRange;
    protected double sumOfDifferenceInRange;
    protected boolean ignoreLeaf;
    protected int ignoreLeafMassThreshold;

    /**
     * A flag that states whether the point to score is known to be contained inside
     * the bounding box of Nodes being accepted. Assumes nodes are accepted in
     * leaf-to-root order.
     */
    protected boolean pointInsideBox;

    /**
     * An array that keeps track of whether each margin of the point being scored is
     * outside inside the box considered during the recursive call to compute the
     * score. Assumes nodes are accepted in leaf-to-root order.
     */
    protected boolean[] coordInsideBox;
    protected IBoundingBoxView shadowBox;

    public AbstractAttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold) {

        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.treeMass = treeMass;
        this.ignoreLeaf = ignoreLeafMassThreshold > DEFAULT_IGNORE_LEAF_MASS_THRESHOLD;
        this.ignoreLeafMassThreshold = ignoreLeafMassThreshold;
        hitDuplicates = false;
        pointInsideBox = false;
        savedScore = 0;
        directionalAttribution = new DiVector(pointToScore.length);
        shadowBox = null;
        coordInsideBox = new boolean[pointToScore.length];
        // array is twice as long as pointToScore because we store
        // positive and negative differences separately
        differenceInRangeVector = new double[2 * pointToScore.length];
    }

    public AbstractAttributionVisitor(float[] pointToScore, int treeMass) {
        this(pointToScore, treeMass, DEFAULT_IGNORE_LEAF_MASS_THRESHOLD);
    }

    /**
     * Take the normalization function applied to the corresponding scoring visitor
     * and apply that to each coordinate of the DiVector to modify the data in
     * place. The function has to be associative in its first parameter; that is, fn
     * (x1, y) + fn (x2, y) = fn (x1 + x2, y)
     * 
     * @return The modified data.
     */
    @Override
    public DiVector getResult() {
        DiVector result = new DiVector(directionalAttribution);
        result.componentwiseTransform(x -> CommonUtils.defaultScalarNormalizerFunction(x, treeMass));
        return result;
    }

    /**
     * Update the anomaly score based on the next step of the tree traversal.
     *
     * @param node        The current node in the tree traversal
     * @param depthOfNode The depth of the current node in the tree
     */
    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }

        IBoundingBoxView smallBox;

        if (hitDuplicates || ignoreLeaf) {
            // use the sibling bounding box to represent counterfactual "what if point & the
            // candidate near neighbor
            // had not been inserted in the tree"

            shadowBox = shadowBox == null ? node.getSiblingBoundingBox(pointToScore)
                    : shadowBox.getMergedBox(node.getSiblingBoundingBox(pointToScore));

            smallBox = shadowBox;
        } else {
            smallBox = node.getBoundingBox();
        }

        IBoundingBoxView largeBox = smallBox.getMergedBox(pointToScore);
        updateRangesForScoring(smallBox, largeBox);

        double probOfCut = sumOfDifferenceInRange / sumOfNewRange;

        // if leaves were ignored we need to keep accounting for the score
        if (ignoreLeaf) {
            savedScore = probOfCut * scoreUnseen(depthOfNode, node.getMass()) + (1 - probOfCut) * savedScore;
        }

        if (probOfCut <= 0) {
            pointInsideBox = true;
        } else {
            double newScore = scoreUnseen(depthOfNode, node.getMass());

            for (int i = 0; i < pointToScore.length; i++) {
                double probOfCutInSpikeDirection = differenceInRangeVector[2 * i] / sumOfNewRange;
                directionalAttribution.high[i] = probOfCutInSpikeDirection * newScore
                        + (1 - probOfCut) * directionalAttribution.high[i];

                double probOfCutInDipDirection = differenceInRangeVector[2 * i + 1] / sumOfNewRange;
                directionalAttribution.low[i] = probOfCutInDipDirection * newScore
                        + (1 - probOfCut) * directionalAttribution.low[i];
            }
        }

        if ((hitDuplicates || ignoreLeaf) && (pointInsideBox || depthOfNode == 0)) {
            // final rescaling; this ensures agreement with the ScalarScoreVector
            // the scoreUnseen/scoreSeen should be the same as scoring; other uses need
            // caution.
            directionalAttribution.renormalize(savedScore);

        }
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {

        updateRangesForScoring(leafNode.getBoundingBox(), leafNode.getBoundingBox().getMergedBox(pointToScore));

        if (Arrays.equals(leafNode.getLeafPoint(), pointToScore)) {
            hitDuplicates = true;
        }

        if ((hitDuplicates) && ((!ignoreLeaf) || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            savedScore = damp(leafNode.getMass(), treeMass) * scoreSeen(depthOfNode, leafNode.getMass());
        } else {
            savedScore = scoreUnseen(depthOfNode, leafNode.getMass());
        }

        if ((hitDuplicates) || ((ignoreLeaf) && (leafNode.getMass() <= ignoreLeafMassThreshold))
                || sumOfNewRange <= 0) {

            Arrays.fill(directionalAttribution.high, savedScore / (2 * pointToScore.length));
            Arrays.fill(directionalAttribution.low, savedScore / (2 * pointToScore.length));
            /* in this case do not have a better option than an equal attribution */
            Arrays.fill(coordInsideBox, false);
        } else {
            for (int i = 0; i < pointToScore.length; i++) {
                directionalAttribution.high[i] = savedScore * differenceInRangeVector[2 * i] / sumOfNewRange;
                directionalAttribution.low[i] = savedScore * differenceInRangeVector[2 * i + 1] / sumOfNewRange;
            }
        }
    }

    /**
     * A scoring function which is applied when the leaf node visited is equal to
     * the point being scored.
     * 
     * @param depth The depth of the node being visited
     * @param mass  The mass of the node being visited
     * @return an anomaly score contribution for a given node
     */
    protected abstract double scoreSeen(int depth, int mass);

    /**
     * A scoring function which is applied when the leaf node visited is not equal
     * to the point being scored. This function is also used to compute the
     * contribution to the anomaly score from non-leaf nodes.
     * 
     * @param depth The depth of the node being visited.
     * @param mass  The mass of the node being visited.
     * @return an anomaly score contribution for a given node.
     */
    protected abstract double scoreUnseen(int depth, int mass);

    /**
     * This function produces a scaling factor which can be used to reduce the
     * influence of leaf nodes with mass greater than 1.
     * 
     * @param leafMass The mass of the leaf node visited
     * @param treeMass The mass of the tree being visited
     * @return a scaling factor to apply to the result from
     *         {@link #scoreSeen(int, int)}.
     */
    protected abstract double damp(int leafMass, int treeMass);

    /**
     * When updating the score for a node, we compare the node's bounding box to the
     * merged bounding box that would be created by adding the point to be scored.
     * This method updates local instance variables sumOfDifferenceInRange and
     * differenceInRange vector to reflect the total difference in side length and
     * the difference in side length in each dimension, respectively.
     *
     * @param smallBox The bounding box corresponding to a Node being visited.
     * @param largeBox The merged bounding box containing smallBox and the point
     *                 being scored.
     */
    protected void updateRangesForScoring(IBoundingBoxView smallBox, IBoundingBoxView largeBox) {
        sumOfDifferenceInRange = 0.0;
        sumOfNewRange = 0.0;
        Arrays.fill(differenceInRangeVector, 0.0);
        for (int i = 0; i < pointToScore.length; i++) {

            sumOfNewRange += largeBox.getRange(i);

            // optimization turned off for ignoreLeaf
            if (coordInsideBox[i] && !ignoreLeaf) {

                continue;
            }

            double maxGap = Math.max(largeBox.getMaxValue(i) - smallBox.getMaxValue(i), 0.0);
            double minGap = Math.max(smallBox.getMinValue(i) - largeBox.getMinValue(i), 0.0);

            if (maxGap + minGap > 0.0) {
                sumOfDifferenceInRange += (minGap + maxGap);
                differenceInRangeVector[2 * i] = maxGap;
                differenceInRangeVector[2 * i + 1] = minGap;

            } else {
                coordInsideBox[i] = true;
            }
        }
    }
}
