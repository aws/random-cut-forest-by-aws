package com.amazon.randomcutforest;

import com.amazon.randomcutforest.anomalydetection.TransductiveScalarScoreVisitor;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Cut;
import com.amazon.randomcutforest.tree.HyperTree;
import com.amazon.randomcutforest.tree.Node;
import com.amazon.randomcutforest.util.NormalMixtureTestData;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Stream;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.defaultScoreSeenFunction;
import static com.amazon.randomcutforest.tree.Cut.isLeftOf;

@Tag("functional")
public class TransductiveHyperForestFunctionalTest {

	private static int numberOfTrees;
	private static int sampleSize;
	private static int dimensions;
	private static int randomSeed;
	private static RandomCutForest parallelExecutionForest;
	private static RandomCutForest singleThreadedForest;
	private static RandomCutForest forestSpy;

	private static double baseMu;
	private static double baseSigma;
	private static double anomalyMu;
	private static double anomalySigma;
	private static double transitionToAnomalyProbability;
	private static double transitionToBaseProbability;
	private static int dataSize;
	private static NormalMixtureTestData generator;
	private static int numTrials = 10;
	private static int numTest = 10;

	public static Function<BoundingBox, double[]> LAlphaSeparation(final double alpha) {

		return (BoundingBox boundingBox) -> {
			double[] answer = new double[boundingBox.getDimensions()];

			for (int i = 0; i < boundingBox.getDimensions(); ++i) {
				double maxVal = boundingBox.getMaxValue(i);
				double minVal = boundingBox.getMinValue(i);
				double oldRange = maxVal - minVal;

				if (oldRange > 0) {
					if (alpha == 0)
						answer[i] = 1.0;
					else
						answer[i] = Math.pow(oldRange, alpha);
				}
			}

			return answer;
		};
	}

	public static Function<BoundingBox, double[]> GTFSeparation(final double gauge) {

		return (BoundingBox boundingBox) -> {
			double[] answer = new double[boundingBox.getDimensions()];

			for (int i = 0; i < boundingBox.getDimensions(); ++i) {
				double maxVal = boundingBox.getMaxValue(i);
				double minVal = boundingBox.getMinValue(i);
				double oldRange = maxVal - minVal;

				if (oldRange > 0) {
					answer[i] = Math.log(1 + oldRange / gauge);
				}
			}

			return answer;
		};
	}

	class HyperForest {
		int dimensions;
		int seed;
		Random random;
		int sampleSize;
		int numberOfTrees;

		ArrayList<HyperTree> trees;

		public HyperForest(int dimensions, int numberOfTrees, int sampleSize, int seed,
				Function<BoundingBox, double[]> vecSeparation) {
			this.numberOfTrees = numberOfTrees;
			this.seed = seed;
			this.sampleSize = sampleSize;
			this.dimensions = dimensions;
			trees = new ArrayList<>();
			random = new Random(seed);
			for (int i = 0; i < numberOfTrees; i++) {
				trees.add(new HyperTree.Builder().buildGVec(vecSeparation).randomSeed(random.nextInt()).build());
			}
		}

		// displacement scoring (multiplied by the normalizer log_2(treesize)) on the
		// fly !!
		// as introduced in Robust Random Cut Forest Based Anomaly Detection in Streams
		// @ICML 2016. This does not address co-displacement (duplicity).
		// seen function is (x,y) -> 1 which basically ignores everything
		// unseen function is (x,y) -> y which corresponds to mass of sibling
		// damp function is (x,y) -> 1 which is no dampening

		public double getDisplacementScore(double[] point) {
			return getDynamicScore(point, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0);
		}
		/*
		 * public double getDisplacementScore(double[] point) { return
		 * getDynamicScoreSim(point, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0); }
		 */
		// Expected height (multiplied by the normalizer log_2(treesize) ) scoring on
		// the fly !!
		// seen function is (x,y) -> x+log(Y)/log(2) which depth + duplicity converted
		// to depth
		// unseen function is (x,y) -> x which is depth
		// damp function is (x,y) -> 1 which is no dampening
		// note that this is *NOT* anything like the expected height in
		// Isolation Forest/Random Forest algorithms, because here
		// the Expected height takes into account the contrafactual
		// that "what would have happened had the point been available during
		// the construction of the forest"

		public double getHeightScore(double[] point) {
			return getDynamicScore(point, (x, y) -> 1.0 * (x + Math.log(y)), (x, y) -> 1.0 * x, (x, y) -> 1.0);
		}

		public double getAnomalyScore(double[] point) {
			return getDynamicScore(point, CommonUtils::defaultScoreSeenFunction,
					CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction);
		}

		public double getDynamicScore(double[] point, BiFunction<Double, Double, Double> seen,
				BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> newDamp) {

			checkArgument(dimensions == point.length, "incorrect dimensions");

			Function<HyperTree, Visitor<Double>> visitorFactory = tree -> new TransductiveScalarScoreVisitor(point,
					tree.getRoot().getMass(), seen, unseen, newDamp, tree.getgVec());
			BinaryOperator<Double> accumulator = Double::sum;

			Function<Double, Double> finisher = sum -> sum / numberOfTrees;

			return trees.parallelStream().map(tree -> {
				Visitor<Double> visitor = visitorFactory.apply(tree);
				return tree.traverseTree(point, visitor);
			}).reduce(accumulator).map(finisher)
					.orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

		}

		void makeForest(double[][] pointList, int prefix) {
			for (int i = 0; i < numberOfTrees; i++) {
				List<double[]> samples = new ArrayList<>();
				boolean[] status = new boolean[pointList.length];
				int y = 0;

				while (y < sampleSize) {
					int z = random.nextInt(prefix);
					if (!status[z]) {
						status[z] = true;
						if (pointList[z].length == dimensions) {
							samples.add(pointList[z]);
						} else {
							throw new IllegalArgumentException("Points have incorrect dimensions");
						}
						++y;
					}
				}
				trees.get(i).makeTree(samples, random.nextInt());
			}
		}

	}
	// ===========================================================

	public static double getSimulatedAnomalyScore(DynamicScoringRandomCutForest forest, double[] point,
			Function<BoundingBox, double[]> gVec) {
		return forest.getDynamicSimulatedScore(point, CommonUtils::defaultScoreSeenFunction,
				CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction, gVec);
	}

	public static double getSimulatedHeightScore(DynamicScoringRandomCutForest forest, double[] point,
			Function<BoundingBox, double[]> gvec) {
		return forest.getDynamicSimulatedScore(point, (x, y) -> 1.0 * (x + Math.log(y)), (x, y) -> 1.0 * x,
				(x, y) -> 1.0, gvec);
	}

	public static double getSimulatedDisplacementScore(DynamicScoringRandomCutForest forest, double[] point,
			Function<BoundingBox, double[]> gvec) {
		return forest.getDynamicSimulatedScore(point, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0, gvec);
	}

	// ===========================================================
	@BeforeAll
	public static void setup() {
		dataSize = 2000;
		numberOfTrees = 100;
		sampleSize = 256;
		dimensions = 30;

		baseMu = 0.0;
		baseSigma = 1.0;
		anomalyMu = 0.0;
		anomalySigma = 1.5;
		transitionToAnomalyProbability = 0.0;
		// ignoring anomaly cluster for now
		transitionToBaseProbability = 1.0;
		generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
				transitionToAnomalyProbability, transitionToBaseProbability);
	}

	private class TestScores {
		double sumCenterScore = 0;
		double sumCenterDisp = 0;
		double sumCenterHeight = 0;
		double sumLeftScore = 0;
		double sumRightScore = 0;
		double sumLeftDisp = 0;
		double sumRightDisp = 0;
		double sumLeftHeight = 0;
		double sumRightHeight = 0;
	}

	public static void runRCF(TestScores testScore, Function<BoundingBox, double[]> gVec) {
		Random prg = new Random(randomSeed);
		for (int trials = 0; trials < numTrials; trials++) {
			double[][] data = generator.generateTestData(dataSize + numTest, dimensions, 100 + trials);
			DynamicScoringRandomCutForest newForest = DynamicScoringRandomCutForest.builder().dimensions(dimensions)
					.numberOfTrees(numberOfTrees).sampleSize(sampleSize).randomSeed(prg.nextInt()).build();

			for (int i = 0; i < dataSize; i++) {
				// shrink, shift at random
				for (int j = 0; j < dimensions; j++)
					data[i][j] *= 0.01;
				if (prg.nextDouble() < 0.5)
					data[i][0] += 5.0;
				else
					data[i][0] -= 5.0;
				newForest.update(data[i]);
				// the points are streamed
			}

			for (int i = dataSize; i < dataSize + numTest; i++) {
				for (int j = 0; j < dimensions; j++)
					data[i][j] *= 0.01;
				testScore.sumCenterScore += getSimulatedAnomalyScore(newForest, data[i], gVec);
				testScore.sumCenterHeight += getSimulatedHeightScore(newForest, data[i], gVec);
				testScore.sumCenterDisp += getSimulatedDisplacementScore(newForest, data[i], gVec);

				data[i][0] += 5; // move to right cluster

				testScore.sumRightScore += getSimulatedAnomalyScore(newForest, data[i], gVec);
				testScore.sumRightHeight += getSimulatedHeightScore(newForest, data[i], gVec);
				testScore.sumRightDisp += getSimulatedDisplacementScore(newForest, data[i], gVec);

				data[i][0] -= 10; // move to left cluster

				testScore.sumLeftScore += getSimulatedAnomalyScore(newForest, data[i], gVec);
				testScore.sumLeftHeight += getSimulatedHeightScore(newForest, data[i], gVec);
				testScore.sumLeftDisp += getSimulatedDisplacementScore(newForest, data[i], gVec);
			}
		}
	}

	public void runGTFLAlpha(TestScores testScore, boolean flag, double gaugeOrAlpha) {
		Random prg = new Random(randomSeed);
		for (int trials = 0; trials < numTrials; trials++) {
			double[][] data = generator.generateTestData(dataSize + numTest, dimensions, 100 + trials);

			HyperForest newForest;
			if (flag)
				newForest = new HyperForest(dimensions, numberOfTrees, sampleSize, prg.nextInt(),
						GTFSeparation(gaugeOrAlpha));
			else
				newForest = new HyperForest(dimensions, numberOfTrees, sampleSize, prg.nextInt(),
						LAlphaSeparation(gaugeOrAlpha));

			for (int i = 0; i < dataSize; i++) {
				// shrink, shift at random
				for (int j = 0; j < dimensions; j++)
					data[i][j] *= 0.01;
				if (prg.nextDouble() < 0.5)
					data[i][0] += 5.0;
				else
					data[i][0] -= 5.0;
			}
			newForest.makeForest(data, dataSize);

			for (int i = dataSize; i < dataSize + numTest; i++) {
				for (int j = 0; j < dimensions; j++)
					data[i][j] *= 0.01;
				testScore.sumCenterScore += newForest.getAnomalyScore(data[i]);
				testScore.sumCenterHeight += newForest.getHeightScore(data[i]);
				testScore.sumCenterDisp += newForest.getDisplacementScore(data[i]);

				data[i][0] += 5; // move to right cluster

				testScore.sumRightScore += newForest.getAnomalyScore(data[i]);
				testScore.sumRightHeight += newForest.getHeightScore(data[i]);
				testScore.sumRightDisp += newForest.getDisplacementScore(data[i]);

				data[i][0] -= 10; // move to left cluster

				testScore.sumLeftScore += newForest.getAnomalyScore(data[i]);
				testScore.sumLeftHeight += newForest.getHeightScore(data[i]);
				testScore.sumLeftDisp += newForest.getDisplacementScore(data[i]);

			}
		}
	}

	public void simulateGTFLAlpha(TestScores testScore, boolean flag, double gaugeOrAlpha) {
		Function<BoundingBox, double[]> gVec = LAlphaSeparation(gaugeOrAlpha);
		if (flag)
			gVec = GTFSeparation(gaugeOrAlpha);
		runRCF(testScore, gVec);
	}

	@Test
	public void GaugeTransductiveForestTest() {

		for (double gauge = 256; gauge >= 1.0 / 1024; gauge /= 4) {
			TestScores testScore = new TestScores();
			runGTFLAlpha(testScore, true, gauge);

			System.out.println();
			System.out.println(" GTF_Gauge =" + gauge + "  ========= ");
			System.out.println("Left Score:    " + testScore.sumLeftScore / (numTest * numTrials));
			System.out.println("Center Score:  " + testScore.sumCenterScore / (numTest * numTrials));
			System.out.println("Right Score:   " + testScore.sumRightScore / (numTest * numTrials));
			System.out.println("Left Height:   " + testScore.sumLeftHeight / (numTest * numTrials));
			System.out.println("Center Height: " + testScore.sumCenterHeight / (numTest * numTrials));
			System.out.println("Right Height:  " + testScore.sumRightHeight / (numTest * numTrials));
			System.out.println("Left Disp:     " + testScore.sumLeftDisp / (numTest * numTrials));
			System.out.println("Center Disp:   " + testScore.sumCenterDisp / (numTest * numTrials));
			System.out.println("Right Disp:    " + testScore.sumRightDisp / (numTest * numTrials));

			TestScores testScoreB = new TestScores();
			simulateGTFLAlpha(testScoreB, true, gauge);

			System.out.println();
			System.out.println(" GTF_Gauge RCF Sim =" + gauge + "  ========= (not choosing dim) ");
			printInfo(testScore, testScoreB);

		}
	}

	@Test
	public void LAlphaForestTest() {

		double alpha = 0;
		for (int i = 0; i <= 20; i++) {

			if (i > 0) {
				if (alpha < 0.9)
					alpha += 0.1;
				else
					alpha = Math.round(alpha + 1);
			}
			TestScores testScore = new TestScores();
			runGTFLAlpha(testScore, false, alpha);

			System.out.println();
			System.out.println(" L_ALPHA = " + alpha + " ========= ");
			System.out.println("Left Score:    " + testScore.sumLeftScore / (numTest * numTrials));
			System.out.println("Center Score:  " + testScore.sumCenterScore / (numTest * numTrials));
			System.out.println("Right Score:   " + testScore.sumRightScore / (numTest * numTrials));
			System.out.println("Left Height:   " + testScore.sumLeftHeight / (numTest * numTrials));
			System.out.println("Center Height: " + testScore.sumCenterHeight / (numTest * numTrials));
			System.out.println("Right Height:  " + testScore.sumRightHeight / (numTest * numTrials));
			System.out.println("Left Disp:     " + testScore.sumLeftDisp / (numTest * numTrials));
			System.out.println("Center Disp:   " + testScore.sumCenterDisp / (numTest * numTrials));
			System.out.println("Right Disp:    " + testScore.sumRightDisp / (numTest * numTrials));

			TestScores testScoreB = new TestScores();
			simulateGTFLAlpha(testScoreB, false, alpha);

			System.out.println();
			System.out.println(" L_ALPHA RCF Sim = " + alpha + " ========= (not choosing dim)");
			printInfo(testScore, testScoreB);

		}
		// large values signify a larger gap

	}

	void printInfo(TestScores testScore, TestScores testScoreB) {
		System.out.println("Left Score:    " + testScoreB.sumLeftScore / (numTest * numTrials));
		System.out.println("Center Score:  " + testScoreB.sumCenterScore / (numTest * numTrials));
		System.out.println("Right Score:   " + testScoreB.sumRightScore / (numTest * numTrials));
		System.out.println("Left Height:   " + testScoreB.sumLeftHeight / (numTest * numTrials));
		System.out.println("Center Height: " + testScoreB.sumCenterHeight / (numTest * numTrials));
		System.out.println("Right Height:  " + testScoreB.sumRightHeight / (numTest * numTrials));
		System.out.println("Left Disp:     " + testScoreB.sumLeftDisp / (numTest * numTrials));
		System.out.println("Center Disp:   " + testScoreB.sumCenterDisp / (numTest * numTrials));
		System.out.println("Right Disp:    " + testScoreB.sumRightDisp / (numTest * numTrials));

		double scoreGap = (testScoreB.sumLeftScore + testScoreB.sumRightScore)
				/ (testScore.sumLeftScore + testScore.sumRightScore);

		System.out.println("Score Gap      " + scoreGap);
		System.out.println("Diff Gap       "
				+ (testScore.sumLeftScore - testScore.sumRightScore)
						/ (testScore.sumLeftScore + testScore.sumRightScore)
				+ " " + (testScoreB.sumLeftScore - testScoreB.sumRightScore)
						/ (testScoreB.sumLeftScore + testScoreB.sumRightScore));

		System.out.println("Center Gap     " + testScoreB.sumCenterScore / (testScore.sumCenterScore * scoreGap));

		double heightGap = (testScoreB.sumLeftHeight + testScoreB.sumRightHeight)
				/ (testScore.sumLeftHeight + testScore.sumRightHeight);

		System.out.println("Height Gap     " + heightGap);
		System.out.println("Diff Gap       "
				+ (testScore.sumLeftHeight - testScore.sumRightHeight)
						/ (testScore.sumLeftHeight + testScore.sumRightHeight)
				+ " " + (testScoreB.sumLeftHeight - testScoreB.sumRightHeight)
						/ (testScoreB.sumLeftHeight + testScoreB.sumRightHeight));

		System.out.println("Center Gap     " + testScoreB.sumCenterHeight / (testScore.sumCenterHeight * heightGap));

		double dispGap = (testScoreB.sumLeftDisp + testScoreB.sumRightDisp)
				/ (testScore.sumLeftDisp + testScore.sumRightDisp);

		System.out.println("Disp Gap       " + dispGap);
		System.out.println("Diff Gap       "
				+ (testScore.sumLeftDisp - testScore.sumRightDisp) / (testScore.sumLeftDisp + testScore.sumRightDisp)
				+ " " + (testScoreB.sumLeftDisp - testScoreB.sumRightDisp)
						/ (testScoreB.sumLeftDisp + testScoreB.sumRightDisp));

		System.out.println("Center Gap     " + testScoreB.sumCenterDisp / (testScore.sumCenterDisp * dispGap));
	}

	@Test
	public void RandomCutForestTest() {

		TestScores testScore = new TestScores();
		runRCF(testScore, LAlphaSeparation(1.0));

		System.out.println();
		System.out.println(" RCF  ========= ");
		System.out.println("Left Score:    " + testScore.sumLeftScore / (numTest * numTrials));
		System.out.println("Center Score:  " + testScore.sumCenterScore / (numTest * numTrials));
		System.out.println("Right Score:   " + testScore.sumRightScore / (numTest * numTrials));
		System.out.println("Left Height:   " + testScore.sumLeftHeight / (numTest * numTrials));
		System.out.println("Center Height: " + testScore.sumCenterHeight / (numTest * numTrials));
		System.out.println("Right Height:  " + testScore.sumRightHeight / (numTest * numTrials));
		System.out.println("Left Disp:     " + testScore.sumLeftDisp / (numTest * numTrials));
		System.out.println("Center Disp:   " + testScore.sumCenterDisp / (numTest * numTrials));
		System.out.println("Right Disp:    " + testScore.sumRightDisp / (numTest * numTrials));

	}

}
