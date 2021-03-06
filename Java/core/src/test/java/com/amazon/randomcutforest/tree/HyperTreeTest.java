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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.DynamicScoringRandomCutForest;
import com.amazon.randomcutforest.VisitorFactory;
import com.amazon.randomcutforest.anomalydetection.TransductiveScalarScoreVisitor;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class HyperTreeTest {

    private static int numberOfTrees;
    private static int sampleSize;
    private static int dimensions;
    private static int randomSeed;

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;
    private static NormalMixtureTestData generator;
    private static int numTrials = 5;
    private static int numTest = 5;

    public static Function<IBoundingBoxView, double[]> LAlphaSeparation(final double alpha) {

        return (IBoundingBoxView boundingBox) -> {
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

    public static Function<IBoundingBoxView, double[]> GTFSeparation(final double gauge) {

        return (IBoundingBoxView boundingBox) -> {
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
                Function<IBoundingBoxView, double[]> vecSeparation) {
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

            VisitorFactory<Double> visitorFactory = new VisitorFactory<>(
                    (tree, y) -> new TransductiveScalarScoreVisitor(tree.projectToTree(y), tree.getMass(), seen, unseen,
                            newDamp, ((HyperTree) tree).getgVec()));
            BinaryOperator<Double> accumulator = Double::sum;

            Function<Double, Double> finisher = sum -> sum / numberOfTrees;

            return trees.parallelStream().map(tree -> tree.traverse(point, visitorFactory)).reduce(accumulator)
                    .map(finisher).orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

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
            Function<IBoundingBoxView, double[]> gVec) {
        return forest.getDynamicSimulatedScore(point, CommonUtils::defaultScoreSeenFunction,
                CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction, gVec);
    }

    public static double getSimulatedHeightScore(DynamicScoringRandomCutForest forest, double[] point,
            Function<IBoundingBoxView, double[]> gvec) {
        return forest.getDynamicSimulatedScore(point, (x, y) -> 1.0 * (x + Math.log(y)), (x, y) -> 1.0 * x,
                (x, y) -> 1.0, gvec);
    }

    public static double getSimulatedDisplacementScore(DynamicScoringRandomCutForest forest, double[] point,
            Function<IBoundingBoxView, double[]> gvec) {
        return forest.getDynamicSimulatedScore(point, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0, gvec);
    }

    // ===========================================================
    @BeforeAll
    public static void setup() {
        dataSize = 2000;
        numberOfTrees = 1; // this is a tree test
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

    public static void runRCF(TestScores testScore, Function<IBoundingBoxView, double[]> gVec) {
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
        assert (testScore.sumCenterScore > 2 * testScore.sumLeftScore);
        assert (testScore.sumCenterScore > 2 * testScore.sumRightScore);

        assert (testScore.sumCenterDisp > 10 * testScore.sumLeftDisp);
        assert (testScore.sumCenterDisp > 10 * testScore.sumRightDisp);

        assert (2 * testScore.sumCenterHeight < testScore.sumLeftHeight);
        assert (2 * testScore.sumCenterHeight < testScore.sumRightHeight);

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

        assert (testScore.sumCenterScore > 1.5 * testScore.sumLeftScore);
        assert (testScore.sumCenterScore > 1.5 * testScore.sumRightScore);

        assert (testScore.sumCenterDisp > 10 * testScore.sumLeftDisp);
        assert (testScore.sumCenterDisp > 10 * testScore.sumRightDisp);

        assert (1.5 * testScore.sumCenterHeight < testScore.sumLeftHeight);
        assert (1.5 * testScore.sumCenterHeight < testScore.sumRightHeight);
    }

    public void simulateGTFLAlpha(TestScores testScore, boolean flag, double gaugeOrAlpha) {
        Function<IBoundingBoxView, double[]> gVec = LAlphaSeparation(gaugeOrAlpha);
        if (flag)
            gVec = GTFSeparation(gaugeOrAlpha);
        runRCF(testScore, gVec);
    }

    @Test
    public void GaugeTransductiveForestTest() {

        TestScores testScoreA = new TestScores();
        runGTFLAlpha(testScoreA, true, 1);
        TestScores testScoreB = new TestScores();
        simulateGTFLAlpha(testScoreB, true, 1);

    }

    @Test
    public void LAlphaForestTest() {

        TestScores testScoreA = new TestScores();
        runGTFLAlpha(testScoreA, false, 0.5);
        TestScores testScoreB = new TestScores();
        simulateGTFLAlpha(testScoreB, false, 0.5);
    }

}
