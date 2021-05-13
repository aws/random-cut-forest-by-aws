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


import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.tree.IBoundingBoxView;
import com.amazon.randomcutforest.tree.INodeView;
import java.util.Arrays;

/**
 * This abstract visitor encodes a standard method for computing a scalar result value. The basic
 * computation is as follows:
 *
 * <ol>
 *   <li>After following the traversal path to a leaf, compute a base score at the leaf node.
 *   <li>For each node in the traversal path from the leaf to the root, compute the probability that
 *       a random cut would separate the query point from the node. The updated score uses this
 *       probability to create a weighted combination between the current score and a score
 *       contribution from the current node.
 * </ol>
 *
 * <p>While this basic algorithm produces good results when all the points in the sample are
 * distinct, it can produce unexpected results when a significant portion of the points in the
 * sample are duplicates. Therefore this class supports different optional features for modifying
 * the score produced when the point being scored is equal to the leaf node in the traversal.
 */
public abstract class AbstractScalarScoreVisitor implements Visitor<Double> {

    public static final int DEFAULT_IGNORE_LEAF_MASS_THRESHOLD = 0;

    /** The point whose anomaly score is being computed. */
    protected final double[] pointToScore;

    /** The mass of the tree being visited. This value is used to normalize the final result. */
    protected final int treeMass;

    /**
     * This flag is set to 'true' if the point being scored is found to be contained by a bounding
     * box in the traversal path, allowing us to short-circuit further computation.
     */
    protected boolean pointInsideBox;

    /**
     * Similar to pointInsideBox, the array coordInsideBox keeps track of whether each coordinate is
     * contained in the corresponding bounding box projection for a bounding box in the traversal
     * path. This field is used to skip unnecessary steps in the probability computation.
     */
    protected boolean[] coordInsideBox;

    /** shadowbox used in attribution and ignoring the leaf to simulate a deletion */
    protected IBoundingBoxView shadowBox = null;

    /**
     * The function used to compute the base score in the case where the point being scored is equal
     * to the leaf point (provided the ignoreLeafEquals and ignoreLeafMassThreshold variables
     * indicate that we should use this method).
     *
     * <p>Function arguments: leaf depth, leaf mass
     */
    protected double score;

    /**
     * If true, then the scoreUnseen method will be used to score a point equal to a leaf point in
     * {@link #acceptLeaf(INodeView, int)}.
     */
    protected boolean ignoreLeafEquals;

    /**
     * If the point being scored is equal to the leaf point but the leaf mass is smaller than this
     * value, then the scoreUnseen method will be used to score the point in {@link
     * #accept(INodeView, int)}.
     */
    protected int ignoreLeafMassThreshold;

    /**
     * Construct a new ScalarScoreVisitor
     *
     * @param pointToScore The point whose anomaly score we are computing
     * @param treeMass The total mass of the RandomCutTree that is scoring the point
     * @param ignoreLeafMassThreshold Is the maximum mass of the leaf which can be ignored
     */
    public AbstractScalarScoreVisitor(
            double[] pointToScore, int treeMass, int ignoreLeafMassThreshold) {
        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.treeMass = treeMass;
        pointInsideBox = false;
        score = 0.0;
        this.ignoreLeafEquals = (ignoreLeafMassThreshold > DEFAULT_IGNORE_LEAF_MASS_THRESHOLD);
        this.ignoreLeafMassThreshold = ignoreLeafMassThreshold;

        // will be initialized to an array of false values
        coordInsideBox = new boolean[pointToScore.length];
    }

    /**
     * Construct a new AbstractScalarScoreVisitor using default leaf options.
     *
     * @param pointToScore The point whose anomaly score we are computing
     * @param treeMass The total mass of the RandomCutTree that is scoring the point
     */
    public AbstractScalarScoreVisitor(double[] pointToScore, int treeMass) {
        this(pointToScore, treeMass, DEFAULT_IGNORE_LEAF_MASS_THRESHOLD);
    }

    /** @return The score computed up until this point. */
    @Override
    public Double getResult() {
        return CommonUtils.defaultScalarNormalizerFunction(score, treeMass);
    }

    /**
     * Update the anomaly score based on the next step of the tree traversal.
     *
     * @param node The current node in the tree traversal
     * @param depthOfNode The depth of the current node in the tree
     */
    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }
        double probabilityOfSeparation;
        if (!ignoreLeafEquals) {
            probabilityOfSeparation = getProbabilityOfSeparation(node.getBoundingBox());
            if (probabilityOfSeparation <= 0) {
                pointInsideBox = true;
                return;
            }
        } else {
            shadowBox =
                    shadowBox == null
                            ? node.getSiblingBoundingBox(pointToScore)
                            : shadowBox.getMergedBox(node.getSiblingBoundingBox(pointToScore));
            probabilityOfSeparation =
                    (shadowBox.getRangeSum() <= 0) ? 1.0 : getProbabilityOfSeparation(shadowBox);
        }

        score =
                probabilityOfSeparation * scoreUnseen(depthOfNode, node.getMass())
                        + (1 - probabilityOfSeparation) * score;
    }

    /**
     * Update the anomaly score with the given leaf node.
     *
     * @param leafNode The leaf node that was reached by traversing the tree
     * @param depthOfNode The depth of the leaf node
     */
    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        if (leafNode.leafPointEquals(pointToScore)
                && (!ignoreLeafEquals || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            pointInsideBox = true;
            score = damp(leafNode.getMass(), treeMass) * scoreSeen(depthOfNode, leafNode.getMass());

        } else {
            score = scoreUnseen(depthOfNode, leafNode.getMass());
        }
    }

    /**
     * A scoring function which is applied when the leaf node visited is equal to the point being
     * scored.
     *
     * @param depth The depth of the node being visited
     * @param mass The mass of the node being visited
     * @return an anomaly score contribution for a given node
     */
    protected abstract double scoreSeen(int depth, int mass);

    /**
     * A scoring function which is applied when the leaf node visited is not equal to the point
     * being scored. This function is also used to compute the contribution to the anomaly score
     * from non-leaf nodes.
     *
     * @param depth The depth of the node being visited.
     * @param mass The mass of the node being visited.
     * @return an anomaly score contribution for a given node.
     */
    protected abstract double scoreUnseen(int depth, int mass);

    /**
     * This function produces a scaling factor which can be used to reduce the influence of leaf
     * nodes with mass greater than 1.
     *
     * @param leafMass The mass of the leaf node visited
     * @param treeMass The mass of the tree being visited
     * @return a scaling factor to apply to the result from {@link #scoreSeen(int, int)}.
     */
    protected abstract double damp(int leafMass, int treeMass);

    /**
     * Compute the probability that a random cut would separate the point from the rest of the
     * bounding box. This method is intended to compute the probability for a non-leaf Node, and
     * will throw an exception if a leaf-node bounding box is detected.
     *
     * @param boundingBox The bounding box that we are computing the probability of separation from.
     * @return is the probability
     */
    protected double getProbabilityOfSeparation(final IBoundingBoxView boundingBox) {
        double sumOfNewRange = 0d;
        double sumOfDifferenceInRange = 0d;

        for (int i = 0; i < pointToScore.length; ++i) {
            double maxVal = boundingBox.getMaxValue(i);
            double minVal = boundingBox.getMinValue(i);
            double oldRange = maxVal - minVal;

            if (!coordInsideBox[i]) {
                if (maxVal < pointToScore[i]) {
                    maxVal = pointToScore[i];
                } else if (minVal > pointToScore[i]) {
                    minVal = pointToScore[i];
                } else if (!ignoreLeafEquals) {
                    // optimization turned on for ignoreLeafEquals==false
                    sumOfNewRange += oldRange;
                    coordInsideBox[i] = true;
                    continue;
                }

                double newRange = maxVal - minVal;
                sumOfNewRange += newRange;
                sumOfDifferenceInRange += (newRange - oldRange);
            } else {
                sumOfNewRange += oldRange;
            }
        }

        if (sumOfNewRange <= 0) {
            // Sum of range across dimensions should only be 0 at leaf nodes as non-leaf
            // nodes always contain
            // more than one distinct point
            throw new IllegalStateException(
                    "Sum of new range of merged box in scoring function is smaller than 0 "
                            + "for a non-leaf node. The sum of range of new bounding box is: "
                            + sumOfNewRange);
        }

        return sumOfDifferenceInRange / sumOfNewRange;
    }
}
