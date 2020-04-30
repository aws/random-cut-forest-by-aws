package com.amazon.randomcutforest.anomalydetection;

import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Node;

public class SimulatedTransductiveScalarScoreVisitor extends TransductiveScalarScoreVisitor {

	private final Function<BoundingBox, double[]> vecSepBuild;

	/**
	 * Construct a new SimulatedTransductiveScalarScoreVisitor
	 *
	 * @param pointToScore
	 *            The point whose anomaly score we are computing
	 * @param treeMass
	 *            The total mass of the RandomCutTree that is scoring the point
	 * @param scoreSeen
	 *            is the part of the score function when the point has been seen
	 * @param scoreUnseen
	 *            is the part of the score when the point has not been seen
	 * @param damp
	 *            corresponds to the dampening of the effect of the seen points
	 * @param vecSepBuild
	 *            A function that provides the probabilities of choosing different
	 *            dimensions given a BoundingBox when the tree was built.
	 * @param vecSepScore
	 *            A function that corresponds to importance of dimensions during
	 *            scoring
	 */
	public SimulatedTransductiveScalarScoreVisitor(double[] pointToScore, int treeMass,
			BiFunction<Double, Double, Double> scoreSeen, BiFunction<Double, Double, Double> scoreUnseen,
			BiFunction<Double, Double, Double> damp, Function<BoundingBox, double[]> vecSepBuild,
			Function<BoundingBox, double[]> vecSepScore) {
		super(pointToScore, treeMass, scoreSeen, scoreUnseen, damp, vecSepScore);
		this.vecSepBuild = vecSepBuild;
	}

	/**
	 * Update the anomaly score based on the next step of the tree traversal.
	 *
	 * @param node
	 *            The current node in the tree traversal
	 * @param depthOfNode
	 *            The depth of the current node in the tree
	 */
	@Override
	public void accept(Node node, int depthOfNode) {
		double weight = getWeight(node.getCut().getDimension(), vecSepBuild, node.getBoundingBox());

		if (pointInsideBox) {
			score *= weight;
			return;
		}

		double probabilityOfSeparation = getProbabilityOfSeparation(node.getBoundingBox());
		if (probabilityOfSeparation == 0) {
			pointInsideBox = true;
		}

		score = probabilityOfSeparation * scoreUnseen(depthOfNode, node.getMass()) + weight * score;

	}

	// The above function differs from TransductiveScalarScoreVisitor only in the
	// weight
	// computation and when the weight function is used.

}
