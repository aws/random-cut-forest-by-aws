package com.amazon.randomcutforest.anomalydetection;

import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Node;

public class TransductiveScalarScoreVisitor extends DynamicScoreVisitor {

    /*
     * the goal of this visitor is to allow tranductive inference; where during
     * scoring we make adjustments so that it appears (to the best of simulation
     * ability) that the tree was built using the knowledge of the point being
     * scored
     * 
     */
    protected final Function<BoundingBox, double[]> vecSepScore;

    /**
     * Construct a new SimulatedTransductiveScalarScoreVisitor
     *
     * @param pointToScore The point whose anomaly score we are computing
     * @param treeMass     The total mass of the RandomCutTree that is scoring the
     *                     point
     * @param scoreSeen    is the part of the score function when the point has been
     *                     seen
     * @param scoreUnseen  is the part of the score when the point has not been seen
     * @param damp         corresponds to the dampening of the effect of the seen
     *                     points
     * @param vecSep       A function that provides the probabilities of choosing
     *                     different dimensions given a BoundingBox when the tree
     *                     was built. This must be the same as the probabilies of
     *                     Transductive inference during scoring. For extenstions
     *                     where these are different, see
     *                     SimulatedTransductiveScalarScoreVisitor
     *
     *                     Note that scores are not normalized because the function
     *                     ranges are unknown as is the case with
     *                     DynamicScoreVisitor
     */

    public TransductiveScalarScoreVisitor(double[] pointToScore, int treeMass,
            BiFunction<Double, Double, Double> scoreSeen, BiFunction<Double, Double, Double> scoreUnseen,
            BiFunction<Double, Double, Double> damp, Function<BoundingBox, double[]> vecSep) {
        super(pointToScore, treeMass, 0, scoreSeen, scoreUnseen, damp);
        this.vecSepScore = vecSep;
        // build function is the same as scoring function
    }

    /**
     * Update the anomaly score based on the next step of the tree traversal.
     *
     * @param node        The current node in the tree traversal
     * @param depthOfNode The depth of the current node in the tree
     */
    @Override
    public void accept(Node node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }
        // note that score was unchanged before the return
        // this is only reasonable if the scoring was done using the same
        // probability function used to build the trees.

        double probabilityOfSeparation = getProbabilityOfSeparation(node.getBoundingBox());
        double weight = getWeight(node.getCut().getDimension(), vecSepScore, node.getBoundingBox());
        if (probabilityOfSeparation == 0) {
            pointInsideBox = true;
            return;
        }

        score = probabilityOfSeparation * scoreUnseen(depthOfNode, node.getMass()) + weight * score;

    }

    /**
     * Compute the probability that a random cut would separate the point from the
     * rest of the bounding box. This method is intended to compute the probability
     * for a non-leaf Node, and will throw an exception if a leaf-node bounding box
     * is detected.
     *
     * @param boundingBox The bounding box that we are computing the probability of
     *                    separation from.
     * @return is the probability
     */
    @Override
    protected double getProbabilityOfSeparation(final BoundingBox boundingBox) {
        double sumOfDenominator = 0d;
        double sumOfNumerator = 0d;

        double[] vec = vecSepScore.apply(boundingBox.getMergedBox(pointToScore));

        for (int i = 0; i < pointToScore.length; ++i) {
            double maxVal = boundingBox.getMaxValue(i);
            double minVal = boundingBox.getMinValue(i);
            double oldRange = maxVal - minVal;
            sumOfDenominator += vec[i];
            if (!coordInsideBox[i]) {
                if (maxVal < pointToScore[i]) {
                    maxVal = pointToScore[i];
                } else if (minVal > pointToScore[i]) {
                    minVal = pointToScore[i];
                }

                double newRange = maxVal - minVal;
                if (newRange > oldRange) {
                    sumOfNumerator += vec[i] * (newRange - oldRange) / newRange;
                } else
                    coordInsideBox[i] = true;
            }
        }

        if (sumOfDenominator <= 0) {
            // Sum of range across dimensions should only be 0 at leaf nodes as non-leaf
            // nodes always contain
            // more than one distinct point
            throw new IllegalStateException("Incorrect State");
        }
        return sumOfNumerator / sumOfDenominator;
        // for RCFs vec[i] = newRange (for dimension i) and therefore the
        // sumOfNumerator is the sum of the difference (after and before
        // merging the point to the box) of ranges
        // sum of denominator is the sum the ranges in each dimension
    }

    // for this visitor class the assumption is that the trees are built using the
    // same probabilities as are used in scoring. In the application herein
    // vecSepBuild
    // is the same as vecSepScore as in the accept(node) above; however the function
    // is
    // written in the more general form so that it can be used for the Simulated
    // version as well without any changes.

    protected double getWeight(int dim, Function<BoundingBox, double[]> vecSepBuild, final BoundingBox boundingBox) {

        double[] vecSmall = vecSepBuild.apply(boundingBox);
        // the smaller box was built!
        BoundingBox largeBox = boundingBox.getMergedBox(pointToScore);
        double[] vecLarge = vecSepScore.apply(largeBox);
        // the larger box is only scored!
        double sumSmall = 0;
        double sumLarge = 0;
        for (int i = 0; i < pointToScore.length; i++) {
            sumSmall += vecSmall[i];
            sumLarge += vecLarge[i];
        }

        return (boundingBox.getRange(dim) / largeBox.getRange(dim)) * (sumSmall / sumLarge)
                * (vecLarge[dim] / vecSmall[dim]);
        // this can be larger than 1
        // For RCFs vecLarge[dim] = largeBox.getRange(dim) and
        // vecSmall[dim] = smallBox.getRange(dim)
        // sumSmall/sumLarge is the probability of non-separation

    }
}
