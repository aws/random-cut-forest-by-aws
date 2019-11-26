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

package com.amazon.randomcutforest.interpolation;

import java.util.Arrays;

import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Node;

import static com.amazon.randomcutforest.tree.Node.isLeftOf;

/**
 * A Visitor which computes several geometric measures that related a given query point to the points stored in a
 * RandomCutTree.
 **/
public class SimpleInterpolationVisitor implements Visitor<InterpolationMeasure> {

    private final double[] pointToScore;
    private final long sampleSize;
    private final boolean centerOfMass;
    public InterpolationMeasure stored;
    double sumOfNewRange = 0d;
    double sumOfDifferenceInRange = 0d;
    double[] directionalDistanceVector;
    double[] differenceInRangeVector;
    /**
     * A flag that states whether the point to score is known to be contained inside the bounding box of Nodes being
     * accepted. Assumes nodes are accepted in leaf-to-root order.
     */
    boolean pointInsideBox;
    /**
     * An array that keeps track of whether each margin of the point being scored is outside inside the box considered
     * during the recursive call to compute the score. Assumes nodes are accepted in leaf-to-root order.
     */
    boolean[] coordInsideBox;
    private boolean pointEqualsLeaf;
    private BoundingBox theShadowBox;
    private double savedMass;
    private double pointMass;

    /**
     * Construct a new Visitor
     *
     * @param pointToScore The point whose anomaly score we are computing
     * @param sampleSize   The sub-sample size used by the RandomCutTree that is scoring the point
     * @param pointMass    indicates the mass/duplicity of the current point
     * @param centerOfMass indicates if the tree has centerOfMass
     */
    public SimpleInterpolationVisitor(double[] pointToScore, int sampleSize, double pointMass, boolean centerOfMass) {
        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.sampleSize = sampleSize;
        // the samplesize may be useful to scale
        pointInsideBox = false;
        this.pointMass = pointMass;    // this corresponds to the mass/duplicity of the query
        stored = new DensityOutput(pointToScore.length, sampleSize);
        directionalDistanceVector = new double[2 * pointToScore.length];
        differenceInRangeVector = new double[2 * pointToScore.length];
        pointEqualsLeaf = false;
        this.centerOfMass = centerOfMass;
        // will be initialized to an array of false values
        coordInsideBox = new boolean[pointToScore.length];
    }

    /**
     * @return The score computed up until this point.
     */
    @Override
    public InterpolationMeasure getResult() {
        return stored;
    }

    @Override
    public void accept(Node node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }
        BoundingBox largeBox;
        BoundingBox smallBox;

        if (pointEqualsLeaf) {
            largeBox = node.getBoundingBox();

            // use the sibling bounding box to represent counterfactual "what if point & the candidate near neighbor
            // had not been inserted in the tree"
            Node sibling = isLeftOf(pointToScore, node) ? node.getRightChild() : node.getLeftChild();
            theShadowBox = theShadowBox == null ? sibling.getBoundingBox() :
                theShadowBox.getMergedBox(sibling.getBoundingBox());
            smallBox = theShadowBox;
        } else {
            smallBox = node.getBoundingBox();
            largeBox = smallBox.getMergedBox(pointToScore);
        }

        updateForCompute(smallBox, largeBox);

        double probOfCut = sumOfDifferenceInRange / sumOfNewRange;
        if (probOfCut <= 0) {
            pointInsideBox = true;
        } else {
            double fieldVal = fieldExt(node, centerOfMass, savedMass, pointToScore);
            double influenceVal = influenceExt(node, centerOfMass, savedMass, pointToScore);
            // if center of mass has been enabled, then those can be used in a similar situation
            // otherwise the center of mass is the 0 vector
            for (int i = 0; i < pointToScore.length; i++) {
                double prob = differenceInRangeVector[2 * i] / sumOfNewRange;
                stored.probMass.high[i] = prob * influenceVal + (1 - probOfCut) * stored.probMass.high[i];
                stored.measure.high[i] = prob * fieldVal
                    + (1 - probOfCut) * stored.measure.high[i];
                stored.distances.high[i] = prob * directionalDistanceVector[2 * i] * influenceVal
                    + (1 - probOfCut) * stored.distances.high[i];

            }
            for (int i = 0; i < pointToScore.length; i++) {
                double prob = differenceInRangeVector[2 * i + 1] / sumOfNewRange;
                stored.probMass.low[i] = prob * influenceVal + (1 - probOfCut) * stored.probMass.low[i];
                stored.measure.low[i] = prob * fieldVal
                    + (1 - probOfCut) * stored.measure.low[i];
                stored.distances.low[i] = prob * directionalDistanceVector[2 * i + 1] * influenceVal
                    + (1 - probOfCut) * stored.distances.low[i];

            }

        }
    }

    @Override
    public void acceptLeaf(Node leafNode, int depthOfNode) {
        updateForCompute(leafNode.getBoundingBox(), leafNode.getBoundingBox().getMergedBox(pointToScore));

        if (sumOfDifferenceInRange <= 0) { // values must be equal
            savedMass = pointMass + leafNode.getMass();
            pointEqualsLeaf = true;
            for (int i = 0; i < pointToScore.length; i++) {
                stored.measure.high[i] = stored.measure.low[i] = 0.5 * selfField(leafNode, savedMass) / pointToScore.length;
                stored.probMass.high[i] = stored.probMass.low[i] = 0.5 * selfInfluence(leafNode, savedMass) / pointToScore.length;
            }
            Arrays.fill(coordInsideBox, false);
        } else {
            savedMass = pointMass;
            double fieldVal = fieldPoint(leafNode, savedMass, pointToScore);
            double influenceVal = influencePoint(leafNode, savedMass, pointToScore);
            for (int i = 0; i < pointToScore.length; i++) {
                double prob = differenceInRangeVector[2 * i] / sumOfNewRange;
                stored.probMass.high[i] = prob * influenceVal;
                stored.measure.high[i] = prob * fieldVal;
                stored.distances.high[i] = prob * directionalDistanceVector[2 * i] * influenceVal;
            }
            for (int i = 0; i < pointToScore.length; i++) {
                double prob = differenceInRangeVector[2 * i + 1] / sumOfNewRange;
                stored.probMass.low[i] = prob * influenceVal;
                stored.measure.low[i] = prob * fieldVal;
                stored.distances.low[i] = prob * directionalDistanceVector[2 * i + 1] * influenceVal;
            }
        }
    }

    /**
     * Update instance variables based on the difference between the large box and small box. The values set by this
     * method are used in {@link #accept} and {@link #acceptLeaf} to update the stored density.
     *
     * @param smallBox
     * @param largeBox
     */
    void updateForCompute(BoundingBox smallBox, BoundingBox largeBox) {

        sumOfNewRange = 0d;
        sumOfDifferenceInRange = 0d;
        Arrays.fill(directionalDistanceVector, 0);
        Arrays.fill(differenceInRangeVector, 0);

        for (int i = 0; i < pointToScore.length; ++i) {
            sumOfNewRange += largeBox.getRange(i);
            if (coordInsideBox[i]) {
                continue;
            }

            double maxGap = Math.max(largeBox.getMaxValue(i) - smallBox.getMaxValue(i), 0.0);
            double minGap = Math.max(smallBox.getMinValue(i) - largeBox.getMinValue(i), 0.0);

            if (maxGap + minGap > 0.0) {
                sumOfDifferenceInRange += (minGap + maxGap);
                differenceInRangeVector[2 * i] = maxGap;
                differenceInRangeVector[2 * i + 1] = minGap;
                if (maxGap > 0) {
                    directionalDistanceVector[2 * i] = (maxGap + smallBox.getRange(i));
                } else {
                    directionalDistanceVector[2 * i + 1] = (minGap + smallBox.getRange(i));
                }
            } else {
                coordInsideBox[i] = true;
            }
        }
    }

    /**
     * The functions below can be changed for arbitrary interpolations.
     *
     * @param node/leafNode corresponds to the node in the tree influencing the current point
     * @param centerOfMass  feature flag describing if the center of mass is enabled in tree
     *                      in general this can be used for arbitrary extensions of the node class
     *                      with additional information.
     * @param thisMass      duplicity of query
     * @param thislocation  location of query
     * @return is the value or a 0/1 function -- the functions can be thresholded based of geometric
     * coordinates of the query and the node. Many different Kernels can be expressed in this decomposed manner.
     */

    double fieldExt(Node node, boolean centerOfMass, double thisMass, double[] thislocation) {
        return (node.getMass() + thisMass);
    }

    double influenceExt(Node node, boolean centerOfMass, double thisMass, double[] thislocation) {
        return 1.0;
    }

    double fieldPoint(Node node, double thisMass, double[] thislocation) {
        return (node.getMass() + thisMass);
    }

    double influencePoint(Node node, double thisMass, double[] thislocation) {
        return 1.0;
    }

    double selfField(Node leafNode, double mass) {
        return mass;
    }

    double selfInfluence(Node leafnode, double mass) {
        return 1.0;
    }

}
