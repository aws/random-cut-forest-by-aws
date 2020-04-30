package com.amazon.randomcutforest.tree;
import com.amazon.randomcutforest.sampler.WeightedPoint;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Cut;
import com.amazon.randomcutforest.tree.Node;
import com.amazon.randomcutforest.tree.RandomCutTree;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.tree.Cut.isLeftOf;

public class HyperTree extends RandomCutTree {

	private final Function<BoundingBox, double[]> gVecBuild;

	public Function<BoundingBox, double[]> getgVec() {
		return gVecBuild;
	}

	public static Builder builder() {
		return new Builder();
	}

	protected HyperTree(HyperTree.Builder builder) {
		super(builder);
		this.gVecBuild = builder.gVec;
	}

	public void makeTree(List<double[]> list, int seed) {
		// this function allows a public call, which may be useful someday
		if (list.size() > 0) {
			// dimensions = list.get(0).length;
			root = makeTreeInt(list, seed, 0, this.gVecBuild);
		} else {
			root = null;
		}
	}

	private Node makeTreeInt(List<double[]> pointList, int seed, int level, Function<BoundingBox, double[]> vecBuild) {

		if (pointList.size() == 0)
			return null;

		BoundingBox thisBox = new BoundingBox(pointList.get(0));
		for (int i = 1; i < pointList.size(); i++) {
			thisBox = thisBox.getMergedBox(pointList.get(i));
		}
		if (thisBox.getRangeSum() <= 0) {
			double[] first = pointList.get(0);
			Node x = new Node(first);
			x.setMass(pointList.size());
			return x;
		}

		Random ring = new Random(seed);
		int leftSeed = ring.nextInt();
		int rightSeed = ring.nextInt();
		Cut cut = getCut(thisBox, ring, vecBuild);

		List<double[]> leftList = new ArrayList<>();
		List<double[]> rightList = new ArrayList<>();

		for (int j = 0; j < pointList.size(); j++) {
			if (isLeftOf(pointList.get(j), cut)) {
				leftList.add(pointList.get(j));
			} else
				rightList.add(pointList.get(j));

		}
		Node leftNode = makeTreeInt(leftList, leftSeed, level + 1, vecBuild);
		Node rightNode = makeTreeInt(rightList, rightSeed, level + 1, vecBuild);
		Node thisNode = new Node(leftNode, rightNode, cut, thisBox);
		leftNode.setParent(thisNode);
		rightNode.setParent(thisNode);
		thisNode.setMass(pointList.size());
		return thisNode;
	}

	private Cut getCut(BoundingBox bb, Random ring, Function<BoundingBox, double[]> vecSeparation) {
		Random rng = new Random(ring.nextInt());
		double cutf = rng.nextDouble();
		double dimf = rng.nextDouble();
		int td = -1;
		double rangeSum = 0;
		double[] vector = vecSeparation.apply(bb);
		for (int i = 0; i < bb.getDimensions(); i++) {
			rangeSum += vector[i];
		}

		double breakPoint = dimf * rangeSum;
		for (int i = 0; i < bb.getDimensions(); i++) {
			double range = vector[i];
			if (range > 0) {
				if ((breakPoint > 0) && (breakPoint <= range)) {
					td = i;
				}
				breakPoint -= range;
			}
		}

		checkArgument(td != -1, "Pivot selection failed.");
		return new Cut(td, bb.getMinValue(td) + bb.getRange(td) * cutf);
	}

	@Override
	public void addPoint(WeightedPoint point) {
		// () -> /dev/null
	}

	@Override
	public void deletePoint(WeightedPoint point) {
		// () -> /dev/null
	}

	public static class Builder extends RandomCutTree.Builder<Builder> {
		private Function<BoundingBox, double[]> gVec;

		public Builder buildGVec(Function<BoundingBox, double[]> gVec) {
			this.gVec = gVec;
			return this;
		}

		public HyperTree build() {
			return new HyperTree(this);
		}
	}
}
