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

package com.amazon.randomcutforest.imputation;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.tree.INodeView;

/**
 * A MultiVisitor which imputes missing values in a point. The missing values
 * are first imputed with the corresponding values in the leaf node in the
 * traversal path. Then, when this MultiVisitor is merged with another
 * MultiVisitor, we keep the imputed values with a lower rank, where the rank
 * value is the anomaly score for the imputed point.
 */
public class ImputeVisitor implements MultiVisitor<double[]> {
    private final boolean[] missing;
    private final boolean[] liftedMissing;
    private double[] queryPoint;
    private double[] liftedPoint;
    private double rank;

    /**
     * Create a new ImputeVisitor.
     * 
     * @param liftedPoint          The point with missing values we want to impute
     * @param queryPoint           The projected point in the tree space
     * @param liftedMissingIndexes the original missing indices
     * @param missingIndexes       The indexes of the missing values in the tree
     *                             space
     */
    public ImputeVisitor(double[] liftedPoint, double[] queryPoint, int[] liftedMissingIndexes, int[] missingIndexes) {
        this.liftedPoint = Arrays.copyOf(liftedPoint, liftedPoint.length);
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.missing = new boolean[queryPoint.length];
        this.liftedMissing = new boolean[liftedPoint.length];

        if (missingIndexes == null) {
            missingIndexes = new int[0];
        }

        for (int i = 0; i < missingIndexes.length; i++) {
            checkArgument(0 <= missingIndexes[i] && missingIndexes[i] < queryPoint.length,
                    "Missing value indexes must be between 0 (inclusive) and queryPoint.length (exclusive)");

            missing[missingIndexes[i]] = true;
        }

        for (int i = 0; i < liftedMissingIndexes.length; i++) {
            checkArgument(0 <= liftedMissingIndexes[i] && liftedMissingIndexes[i] < liftedPoint.length,
                    "Missing value indexes must be between 0 (inclusive) and liftedPoint.length (exclusive)");

            liftedMissing[liftedMissingIndexes[i]] = true;
        }

        rank = 10.0;
    }

    public ImputeVisitor(double[] queryPoint, int[] missingIndexes) {
        this(Arrays.copyOf(queryPoint, queryPoint.length), queryPoint,
                Arrays.copyOf(missingIndexes, missingIndexes.length), missingIndexes);
    }

    public ImputeVisitor(double[] queryPoint, int numberOfMissingIndices, int[] missingIndexes) {
        this(queryPoint, Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)));
    }

    /**
     * A copy constructor which creates a deep copy of the original ImputeVisitor.
     *
     * @param original
     */
    ImputeVisitor(ImputeVisitor original) {
        int length = original.queryPoint.length;
        this.queryPoint = Arrays.copyOf(original.queryPoint, length);
        this.missing = Arrays.copyOf(original.missing, length);
        this.liftedPoint = Arrays.copyOf(original.liftedPoint, original.liftedPoint.length);
        this.liftedMissing = Arrays.copyOf(original.liftedMissing, original.liftedPoint.length);
        rank = 10.0;
    }

    /**
     * @return the rank of the imputed point in this visitor.
     */
    public double getRank() {
        return rank;
    }

    /**
     * Update the rank value using the probability that the imputed query point is
     * separated from this bounding box in a random cut. This step is conceptually
     * the same as * {@link AnomalyScoreVisitor#accept}.
     *
     * @param node        the node being visited
     * @param depthOfNode the depth of the node being visited
     */
    public void accept(final INodeView node, final int depthOfNode) {

        double probabilityOfSeparation = CommonUtils.getProbabilityOfSeparation(node.getBoundingBox(), queryPoint);

        if (probabilityOfSeparation <= 0) {
            return;
        }

        rank = probabilityOfSeparation * scoreUnseen(depthOfNode, node.getMass())
                + (1 - probabilityOfSeparation) * rank;
    }

    /**
     * Impute the missing values in the query point with the corresponding values in
     * the leaf point. Set the rank to the score function evaluated at the leaf
     * node.
     *
     * @param leafNode    the leaf node being visited
     * @param depthOfNode the depth of the leaf node
     */
    @Override
    public void acceptLeaf(final INodeView leafNode, final int depthOfNode) {
        double[] leafPoint = leafNode.getLeafPoint();
        for (int i = 0; i < queryPoint.length; i++) {
            if (missing[i]) {
                queryPoint[i] = leafPoint[i];
            }
        }
        double[] liftedLeafPoint = leafNode.getLiftedLeafPoint();
        for (int i = 0; i < liftedLeafPoint.length; i++) {
            if (liftedMissing[i]) {
                liftedPoint[i] = liftedLeafPoint[i];
            }
        }
        double probabilityOfSeparation = CommonUtils.getProbabilityOfSeparation(leafNode.getBoundingBox(), queryPoint);
        if (probabilityOfSeparation <= 0) {
            if (depthOfNode == 0) {
                rank = 0;
            } else {
                rank = scoreSeen(depthOfNode, leafNode.getMass());
            }
        } else {
            rank = scoreUnseen(depthOfNode, leafNode.getMass());
        }
    }

    /**
     * @return the imputed point.
     */
    @Override
    public double[] getResult() {
        return liftedPoint;
    }

    /**
     * An ImputeVisitor should split whenever the cut dimension in a node
     * corresponds to a missing value in the query point.
     *
     * @param node A node in the tree traversal
     * @return true if the cut dimension in the node corresponds to a missing value
     *         in the query point, false otherwise.
     */
    @Override
    public boolean trigger(final INodeView node) {
        return missing[node.getCutDimension()];
    }

    /**
     * @return a copy of this visitor.
     */
    @Override
    public MultiVisitor<double[]> newCopy() {
        return new ImputeVisitor(this);
    }

    /**
     * If this visitor as a lower rank than the second visitor, do nothing.
     * Otherwise, overwrite this visitor's imputed values withe the valuse from the
     * second visitor.
     *
     * @param other A second visitor
     */
    @Override
    public void combine(MultiVisitor<double[]> other) {
        ImputeVisitor visitor = (ImputeVisitor) other;
        if (visitor.getRank() < rank) {
            System.arraycopy(visitor.queryPoint, 0, queryPoint, 0, queryPoint.length);
            System.arraycopy(visitor.liftedPoint, 0, liftedPoint, 0, liftedPoint.length);
            rank = visitor.getRank();
        }
    }

    protected double scoreSeen(int depth, int mass) {
        return CommonUtils.defaultScoreSeenFunction(depth, mass);
    }

    protected double scoreUnseen(int depth, int mass) {
        return CommonUtils.defaultScoreUnseenFunction(depth, mass);
    }
}
