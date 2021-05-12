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

package com.amazon.randomcutforest;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;
import org.junit.jupiter.params.provider.CsvSource;

import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

@Tag("functional")
public class RandomCutForestFunctionalTest {

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

    @BeforeAll
    public static void oneTimeSetUp() { // this is a stochastic dataset and will have different values for different
                                        // runs
        numberOfTrees = 100;
        sampleSize = 256;
        dimensions = 3;
        randomSeed = 123;

        parallelExecutionForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(dimensions).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        singleThreadedForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(dimensions).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).parallelExecutionEnabled(false).build();

        dataSize = 10_000;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 5.0;
        anomalySigma = 1.5;
        transitionToAnomalyProbability = 0.01;
        transitionToBaseProbability = 0.4;

        NormalMixtureTestData generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, dimensions);

        for (int i = 0; i < dataSize; i++) {
            parallelExecutionForest.update(data[i]);
            singleThreadedForest.update(data[i]);
        }
    }

    // Use this ArgumentsProvider to run a test on both single-threaded and
    // multi-threaded forests
    static class TestForestProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {
            return Stream.of(singleThreadedForest, parallelExecutionForest).map(Arguments::of);
        }
    }

    // displacement scoring (multiplied by the normalizer log_2(treesize)) on the
    // fly !!
    // as introduced in Robust Random Cut Forest Based Anomaly Detection in Streams
    // @ICML 2016. This does not address co-displacement (duplicity).
    // seen function is (x,y) -> 1 which basically ignores everything
    // unseen function is (x,y) -> y which corresponds to mass of sibling
    // damp function is (x,y) -> 1 which is no dampening

    public static double getDisplacementScore(DynamicScoringRandomCutForest forest, double[] point) {
        return forest.getDynamicScore(point, 0, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0);
    }

    public double getDisplacementScoreApproximate(DynamicScoringRandomCutForest forest, double[] point,
            double precision) {
        return forest.getApproximateDynamicScore(point, precision, true, 0, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0);
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

    public static double getHeightScore(DynamicScoringRandomCutForest forest, double[] point) {
        return forest.getDynamicScore(point, 0, (x, y) -> 1.0 * (x + Math.log(y)), (x, y) -> 1.0 * x, (x, y) -> 1.0);
    }

    public double getHeightScoreApproximate(DynamicScoringRandomCutForest forest, double[] point, double precision) {
        return forest.getApproximateDynamicScore(point, precision, false, 0, (x, y) -> 1.0 * (x + Math.log(y)),
                (x, y) -> 1.0 * x, (x, y) -> 1.0);
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    private void testGetAnomalyScore(DynamicScoringRandomCutForest forest) {
        double[] point = { 0.0, 0.0, 0.0 };
        double score = forest.getAnomalyScore(point);
        assertTrue(score < 1);
        assertTrue(forest.getApproximateAnomalyScore(point) < 1);

        /**
         * This part demonstrates testing of dynamic scoring where score functions are
         * changed on the fly.
         */

        // displacement scoring on the fly!!

        score = getDisplacementScore(forest, point);
        assertTrue(score < 25);
        // testing that the leaf exclusion does not affect anything
        // tests the masking effect
        assertTrue(forest.getDynamicScore(point, 1, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0) < 25);
        double newScore = getDisplacementScoreApproximate(forest, point, 0);
        assertEquals(score, newScore, 1E-10);
        double otherScore = getDisplacementScoreApproximate(forest, point, 0.1);
        assertTrue(otherScore < 25);
        // the approximation bound is increased to accomodate the
        // larger variance of the probabilistic test
        // adjust the parameters in early convergence to
        // get 0.1*score+0.1
        assertEquals(otherScore, newScore, 0.3 * score + 0.1);

        /**
         * Using expected height -- note that this height is not the same as the height
         * in a random forest, because it accounts for the contrafactual of having
         * constructed the forest with the knowledge of the point.
         */

        score = getHeightScore(forest, point);
        assertTrue(score > 50);
        newScore = getHeightScoreApproximate(forest, point, 0);
        assertEquals(score, newScore, 1E-10);
        otherScore = getHeightScoreApproximate(forest, point, 0.1);
        assertTrue(otherScore > 50);
        // the approximation bound is increased to accomodate the
        // larger variance of the probabilistic test
        assertEquals(score, otherScore, 0.3 * score + 0.1);

        point = new double[] { 8.0, 8.0, 8.0 };
        score = forest.getAnomalyScore(point);
        assertTrue(score > 1);
        assertTrue(forest.getApproximateAnomalyScore(point) > 1);

        // displacement scoring on the fly !!
        score = getDisplacementScore(forest, point);
        assertTrue(score > 100);
        // testing masking
        assertTrue(forest.getDynamicScore(point, 1, (x, y) -> 1.0, (x, y) -> y, (x, y) -> 1.0) > 100);
        newScore = getDisplacementScoreApproximate(forest, point, 0);
        assertEquals(score, newScore, 1E-10);
        otherScore = getDisplacementScoreApproximate(forest, point, 0.1);
        assertTrue(otherScore > 100);
        // the approximation bound is increased to accomodate the
        // larger variance of the probabilistic test
        assertEquals(score, otherScore, 0.3 * score + 0.1);

        // Expected height scoring on the fly !!
        score = getHeightScore(forest, point);
        assertTrue(score < 30);
        newScore = getHeightScoreApproximate(forest, point, 0);
        assertEquals(score, newScore, 1E-10);
        otherScore = getHeightScoreApproximate(forest, point, 0.1);
        assertTrue(otherScore < 30);
        // the approximation bound is increased to accomodate the
        // larger variance of the probabilistic test
        assertEquals(score, otherScore, 0.3 * score + 0.1);

    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testSideEffectsA(RandomCutForest forest) {
        double score = forest.getAnomalyScore(new double[] { 0.0, 0.0, 0.0 });
        NormalMixtureTestData generator2 = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] newData = generator2.generateTestData(dataSize, dimensions);
        for (int i = 0; i < dataSize; i++) {
            forest.getAnomalyScore(newData[i]);
        }
        double newScore = forest.getAnomalyScore(new double[] { 0.0, 0.0, 0.0 });
        assertEquals(score, newScore, 10E-10);
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testSideEffectsB(RandomCutForest forest) {
        /* the changes to score and attribution should be in sync */
        DiVector initial = forest.getAnomalyAttribution(new double[] { 0.0, 0.0, 0.0 });
        NormalMixtureTestData generator2 = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] newData = generator2.generateTestData(dataSize, dimensions);
        for (int i = 0; i < dataSize; i++) {
            forest.getAnomalyAttribution(newData[i]);
        }
        double newScore = forest.getAnomalyScore(new double[] { 0.0, 0.0, 0.0 });
        DiVector newVector = forest.getAnomalyAttribution(new double[] { 0.0, 0.0, 0.0 });
        assertEquals(initial.getHighLowSum(), newVector.getHighLowSum(), 10E-10);
        assertEquals(initial.getHighLowSum(), newScore, 1E-10);
        assertArrayEquals(initial.high, newVector.high, 1E-10);
        assertArrayEquals(initial.low, newVector.low, 1E-10);
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testGetAnomalyAttribution(RandomCutForest forest) {

        /* This method checks that the scores and attributions are consistent */

        double[] point = { 0.0, 0.0, 0.0 };
        DiVector seenResult = forest.getAnomalyAttribution(point);
        double seenScore = forest.getAnomalyScore(point);
        assertTrue(seenResult.getHighLowSum(0) < 0.5);
        assertTrue(seenResult.getHighLowSum(1) < 0.5);
        assertTrue(seenResult.getHighLowSum(2) < 0.5);
        assertTrue(seenScore < 1.0);
        assertEquals(seenScore, seenResult.getHighLowSum(), 1E-10);

        DiVector likelyResult = forest.getApproximateAnomalyAttribution(point);
        double score = forest.getApproximateAnomalyScore(point);
        assertTrue(likelyResult.getHighLowSum(0) < 0.5);
        assertTrue(likelyResult.getHighLowSum(1) < 0.5);
        assertTrue(likelyResult.getHighLowSum(2) < 0.5);
        assertEquals(score, likelyResult.getHighLowSum(), 0.1);
        assertEquals(seenResult.getHighLowSum(), likelyResult.getHighLowSum(), 0.1);
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testMultipleAttributions(RandomCutForest forest) {

        /**
         * We will test the attribution over random runs. Narrow tests can fail -- we
         * will keep track of the aggregate number of narrow tests and test for large
         * characterization that would be misleading in failure.
         */
        int hardPass = 0;
        int causal = 0;
        double[] point = { 6.0, 0.0, 0.0 };
        DiVector result = forest.getAnomalyAttribution(point);
        assertTrue(result.low[0] < 0.2);
        if (result.getHighLowSum(1) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(2) < 0.5)
            ++hardPass;
        assertTrue(result.getHighLowSum(1) + result.getHighLowSum(2) < 1.0);
        assertTrue(result.high[0] > forest.getAnomalyScore(point) / 3);
        if (result.high[0] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        // the last line states that first coordinate was high and was a majority
        // contributor to the score
        // the previous test states that the contribution is twice the average of the 12
        // possible contributors.
        // these tests all subparts of the score at once

        point = new double[] { -6.0, 0.0, 0.0 };
        result = forest.getAnomalyAttribution(point);
        assertTrue(result.getHighLowSum() > 1.0);
        assertTrue(result.high[0] < 0.5);
        if (result.getHighLowSum(1) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(2) < 0.5)
            ++hardPass;
        assertTrue(result.low[0] > forest.getAnomalyScore(point) / 3);
        if (result.low[0] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        point = new double[] { 0.0, 6.0, 0.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        if (result.getHighLowSum(0) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(2) < 0.5)
            ++hardPass;
        assertTrue(result.low[1] < 0.5);
        assertTrue(result.high[1] > forest.getAnomalyScore(point) / 3);
        if (result.high[1] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        point = new double[] { 0.0, -6.0, 0.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        if (result.getHighLowSum(0) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(2) < 0.5)
            ++hardPass;
        assertTrue(result.high[1] < 0.5);
        assertTrue(result.low[1] > forest.getAnomalyScore(point) / 3);
        if (result.low[1] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        point = new double[] { 0.0, 0.0, 6.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        if (result.getHighLowSum(0) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(1) < 0.5)
            ++hardPass;
        assertTrue(result.low[2] < 0.5);
        assertTrue(result.high[2] > forest.getAnomalyScore(point) / 3);
        if (result.high[2] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        point = new double[] { 0.0, 0.0, -6.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        if (result.getHighLowSum(0) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(1) < 0.5)
            ++hardPass;
        assertTrue(result.high[2] < 0.5);
        assertTrue(result.low[2] > forest.getAnomalyScore(point) / 3);
        if (result.low[2] > 0.5 * forest.getAnomalyScore(point))
            ++causal;

        assertTrue(causal >= 5); // maximum is 6; there can be skew in one direction

        point = new double[] { -3.0, 0.0, 0.0 };
        result = forest.getAnomalyAttribution(point);
        assertTrue(result.high[0] < 0.5);
        if (result.getHighLowSum(1) < 0.5)
            ++hardPass;
        if (result.getHighLowSum(2) < 0.5)
            ++hardPass;
        assertTrue(result.low[0] > forest.getAnomalyScore(point) / 3);

        /*
         * For multiple causes, the relationship of scores only hold for larger
         * distances.
         */

        point = new double[] { -3.0, 6.0, 0.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        if (result.low[0] > 0.5)
            ++hardPass;
        assertTrue(result.high[0] < 0.5);
        assertTrue(result.low[1] < 0.5);
        assertTrue(result.high[1] > 0.5);
        if (result.high[1] > 0.9)
            ++hardPass;
        assertTrue(result.getHighLowSum(2) < 0.5);
        assertTrue(result.high[1] + result.low[0] > 0.8 * forest.getAnomalyScore(point));

        point = new double[] { 6.0, -3.0, 0.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        assertTrue(result.low[0] < 0.5);
        assertTrue(result.high[0] > 0.5);
        if (result.high[0] > 0.9)
            ++hardPass;
        if (result.low[1] > 0.5)
            ++hardPass;
        assertTrue(result.high[1] < 0.5);
        assertTrue(result.getHighLowSum(2) < 0.5);
        assertTrue(result.high[0] + result.low[1] > 0.8 * forest.getAnomalyScore(point));

        point = new double[] { 20.0, -10.0, 0.0 };
        assertTrue(result.getHighLowSum() > 1.0);
        result = forest.getAnomalyAttribution(point);
        assertTrue(result.high[0] + result.low[1] > 0.8 * forest.getAnomalyScore(point));
        if (result.high[0] > 1.8 * result.low[1])
            ++hardPass;
        if (result.low[1] > result.high[0] / 2.2)
            ++hardPass;

        assertTrue(hardPass >= 15); // maximum is 20
    }

    @Test
    public void testUpdateWithSignedZeros() {
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(2).dimensions(1)
                .randomSeed(randomSeed).centerOfMassEnabled(true).storeSequenceIndexesEnabled(true).build();

        forest.update(new double[] { 0.0 });
        forest.getAnomalyScore(new double[] { 0.0 });
        forest.getAnomalyScore(new double[] { -0.0 });

        forest.update(new double[] { -0.0 });
        forest.getAnomalyScore(new double[] { 0.0 });
        forest.getAnomalyScore(new double[] { -0.0 });
    }

    @Test
    public void testShadowBuffer() {
        /**
         * This test checks that the attribution *DOES NOT* change as a ratio as more
         * copies of the points are added. The shadowbox in
         * the @DirectionalAttributionVisitor allows us to simulate a deletion without
         * performing a deletion.
         *
         * The goal is to measure the attribution and have many copies of the same point
         * and eventually the attribution will become uniform in all directions.
         *
         * we create a new forest so that other tests are unaffected.
         */
        numberOfTrees = 100;
        sampleSize = 256;
        dimensions = 3;
        randomSeed = 123;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(dimensions).randomSeed(randomSeed).centerOfMassEnabled(true).lambda(1e-5)
                .storeSequenceIndexesEnabled(true).build();

        dataSize = 10_000;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 5.0;
        anomalySigma = 1.5;
        transitionToAnomalyProbability = 0.01;
        transitionToBaseProbability = 0.4;

        NormalMixtureTestData generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, dimensions);

        for (int i = 0; i < dataSize; i++) {
            newForest.update(data[i]);
        }

        double[] point = new double[] { -8.0, -8.0, 0.0 };
        DiVector result = newForest.getAnomalyAttribution(point);
        double score = newForest.getAnomalyScore(point);
        assertEquals(score, result.getHighLowSum(), 1E-10);
        assertTrue(score > 2);
        assertTrue(result.getHighLowSum(2) < 0.2);
        // the third dimension has little influence in classification

        // this is going to add {8,8,0} into the forest
        // but not enough to cause large scale changes
        // note the probability of a tree seeing a change is
        // 256/10_000
        for (int i = 0; i < 5; i++) {
            newForest.update(point);
        }

        DiVector newResult = newForest.getAnomalyAttribution(point);
        double newScore = newForest.getAnomalyScore(point);

        assertEquals(newScore, newResult.getHighLowSum(), 1E-10);
        assertTrue(newScore < score);
        for (int j = 0; j < 3; j++) {
            // relationship holds at larger values
            if (result.high[j] > 0.2) {
                assertEquals(score * newResult.high[j], newScore * result.high[j], 0.1 * score);
            } else {
                assertTrue(newResult.high[j] < 0.2);
            }

            if (result.low[j] > 0.2) {
                assertEquals(score * newResult.low[j], newScore * result.low[j], 0.1 * score);
            } else {
                assertTrue(newResult.low[j] < 0.2);
            }
        }

        // this will make the point an inlier
        for (int i = 0; i < 5000; i++) {
            newForest.update(point);
        }

        DiVector finalResult = newForest.getAnomalyAttribution(point);
        double finalScore = newForest.getAnomalyScore(point);
        assertTrue(finalScore < 1);
        assertEquals(finalScore, finalResult.getHighLowSum(), 1E-10);

        for (int j = 0; j < 3; j++) {
            // relationship holds at larger values
            if (finalResult.high[j] > 0.2) {
                assertEquals(score * finalResult.high[j], finalScore * result.high[j], 0.1 * score);
            } else {
                assertTrue(newResult.high[j] < 0.2);
            }

            if (finalResult.low[j] > 0.2) {
                assertEquals(score * finalResult.low[j], finalScore * result.low[j], 0.1 * score);
            } else {
                assertTrue(finalResult.low[j] < 0.2);
            }
        }

    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testSimpleDensity(RandomCutForest forest) {

        DensityOutput output1 = forest.getSimpleDensity(new double[] { 0.0, 0.0, 0.0 });
        DensityOutput output2 = forest.getSimpleDensity(new double[] { 6.0, 6.0, 0.0 });
        DensityOutput output3 = forest.getSimpleDensity(new double[] { -4.0, -4.0, 0.0 });
        DensityOutput output4 = forest.getSimpleDensity(new double[] { -6.0, -6.0, 0.0 });

        assertTrue(output1.getDensity(0.001, 3) > output2.getDensity(0.001, 3));
        assertTrue(output1.getDensity(0.001, 3) > output3.getDensity(0.001, 3));
        assertTrue(output1.getDensity(0.001, 3) > output4.getDensity(0.001, 3));
        assertTrue(output3.getDensity(0.001, 3) > output4.getDensity(0.001, 3));
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testSimpleDensityWhenSamplerNotFullThenDensityIsZero(RandomCutForest forest) {
        RandomCutForest forestSpy = spy(forest);
        when(forestSpy.samplersFull()).thenReturn(false);

        DensityOutput output = forestSpy.getSimpleDensity(new double[] { 0.0, 0.0, 0.0 });
        assertEquals(0, output.getDensity(0.001, 3));
    }

    @ParameterizedTest
    @ArgumentsSource(TestForestProvider.class)
    public void testImputeMissingValues(RandomCutForest forest) {

        double[] queryPoint = new double[] { Double.NaN, 0.02, 0.01 };
        int numberOfMissingValues = 1;
        int[] missingIndexes = new int[] { 0 };

        double[] imputedPoint = forest.imputeMissingValues(queryPoint, numberOfMissingValues, missingIndexes);
        assertEquals(queryPoint[1], imputedPoint[1]);
        assertTrue(Math.abs(imputedPoint[0]) < 0.5);
    }

    @Test
    public void testExtrapolateA() {

        numberOfTrees = 100;
        sampleSize = 256;
        int shinglesize = 10;
        randomSeed = 123;

        RandomCutForest newforest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        double amplitude = 50.0;
        double noise = 2.0;
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 850;
        double[] data = getDataA(amplitude, noise);
        double[] answer = null;
        double error = 0;
        double[] record = null;
        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforest.update(record);
            }
        }

        answer = newforest.extrapolateBasic(record, 200, 1, false);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double prediction = amplitude * cos((j + 850 - 50) * 2 * PI / 120);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;

        assertTrue(error < 4 * noise);

    }

    @Test
    public void testExtrapolateB() {

        numberOfTrees = 100;
        sampleSize = 256;
        int shinglesize = 120;
        randomSeed = 123;

        RandomCutForest newforestB = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        double amplitude = 50.0;
        double noise = 2.0;
        Random noiseprg = new Random(72);
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 850;
        double[] data = getDataA(amplitude, noise);
        double[] answer = null;
        double error = 0;

        double[] record = null;
        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                // produce cyclic vectors
                record = getShinglePoint(history, 0, shinglesize);
                newforestB.update(record);
            }
        }

        answer = newforestB.extrapolateBasic(record, 200, 1, true, entryIndex);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double prediction = amplitude * cos((j + 850 - 50) * 2 * PI / 120);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 4 * noise);

    }

    @Test
    public void testExtrapolateC() {

        numberOfTrees = 100;
        sampleSize = 256;
        int shinglesize = 20;
        randomSeed = 124;

        // build two identical copies; we will be giving them different
        // subsequent inputs and test adaptation to stream evolution

        RandomCutForest newforestC = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).centerOfMassEnabled(true).lambda(1.0 / 300)
                .storeSequenceIndexesEnabled(true).build();

        RandomCutForest newforestD = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).centerOfMassEnabled(true).lambda(1.0 / 300)
                .storeSequenceIndexesEnabled(true).build();

        double amplitude = 50.0;
        double noise = 2.0;
        Random noiseprg = new Random(72);
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 1330;
        double[] data = getDataB(amplitude, noise);
        double[] answer = null;
        double error = 0;

        double[] record = null;
        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestC.update(record);
                newforestD.update(record);
            }
        }
        /**
         * the two forests are identical up to this point we will now provide two
         * different input to each num+2*expLife=1930, but since the shape of the
         * pattern remains the same in a phase shift, the prediction comes back to
         * "normal" fairly quickly.
         */

        for (int j = num; j < 1630; ++j) { // we stream here ....
            double t = cos(2 * PI * (j - 50) / 240);
            history[entryIndex] = amplitude * t + noise * noiseprg.nextDouble();
            ;
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestC.update(record);
            }
        }
        answer = newforestC.extrapolateBasic(record, 200, 1, false);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double t = cos(2 * PI * (1630 + j - 50) / 240);
            double prediction = amplitude * t;
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 2 * noise);

        /**
         * Here num+2*expLife=1930 for a small explife such as 300, num+expLife is
         * already sufficient increase the factor for larger expLife or increase the
         * sampleSize to absorb the longer range dependencies of a larger expLife
         */
        for (int j = num; j < 1630; ++j) { // we stream here ....
            double t = cos(2 * PI * (j + 50) / 120);
            int sign = (t > 0) ? 1 : -1;
            history[entryIndex] = amplitude * sign * Math.pow(t * sign, 1.0 / 3) + noise * noiseprg.nextDouble();
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestD.update(record);
            }
        }
        answer = newforestD.extrapolateBasic(record, 200, 1, false);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double t = cos(2 * PI * (1630 + j + 50) / 120);
            int sign = (t > 0) ? 1 : -1;
            double prediction = amplitude * sign * Math.pow(t * sign, 1.0 / 3);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 2 * noise);
    }

    @Test
    public void getTotalUpdates_returnExpectedSize() {
        assertEquals(dataSize, singleThreadedForest.getTotalUpdates());
        assertEquals(dataSize, parallelExecutionForest.getTotalUpdates());
    }

    double[] getDataA(double amplitude, double noise) {
        int num = 850;
        double[] data = new double[num];
        Random noiseprg = new Random(9000);

        for (int i = 0; i < 510; i++) {
            data[i] = amplitude * cos(2 * PI * (i - 50) / 120) + noise * noiseprg.nextDouble();
        }
        for (int i = 510; i < 525; i++) { // flatline
            data[i] = 0;
        }
        for (int i = 525; i < 825; i++) {
            data[i] = amplitude * cos(2 * PI * (i - 50) / 120) + noise * noiseprg.nextDouble();
        }
        for (int i = 825; i < num; i++) { // high frequency noise
            data[i] = amplitude * cos(2 * PI * (i - 50) / 12) + noise * noiseprg.nextDouble();
        }
        return data;
    }

    double[] getDataB(double amplitude, double noise) {
        int num = 1330;
        double[] data = new double[num];
        Random noiseprg = new Random(9001);
        for (int i = 0; i < 990; i++) {
            data[i] = amplitude * cos(2 * PI * (i + 50) / 240) + noise * noiseprg.nextDouble();
        }
        for (int i = 990; i < 1005; i++) { // flatline
            data[i] = 0;
        }
        for (int i = 1005; i < 1305; i++) {
            data[i] = amplitude * cos(2 * PI * (i + 50) / 240) + noise * noiseprg.nextDouble();
        }
        for (int i = 1305; i < num; i++) { // high frequency noise
            data[i] = amplitude * cos(2 * PI * (i + 50) / 12) + noise * noiseprg.nextDouble();
        }
        return data;
    }

    private static double[] getShinglePoint(double[] recentPointsSeen, int indexOfOldestPoint, int shingleLength) {
        double[] shingledPoint = new double[shingleLength];
        int i = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            shingledPoint[i++] = point;

        }
        return shingledPoint;
    }

    @ParameterizedTest(name = "{index} => numDims={0}, numTrees={1}, numSamples={2}, numTrainSamples={3}, "
            + "numTestSamples={4}, enableParallel={5}, numThreads={6}")
    @CsvSource({ "10, 50, 256, 50000, 0, 0, 0" })
    public void dynamicCachingChangeTest(int numDims, int numTrees, int numSamples, int numTrainSamples,
            int numTestSamples, int enableParallel, int numThreads) {
        RandomCutForest.Builder<?> forestBuilder = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).boundingBoxCacheFraction(1.0).compact(false);
        if (enableParallel == 0) {
            forestBuilder.parallelExecutionEnabled(false);
        }
        if (numThreads > 0) {
            forestBuilder.threadPoolSize(numThreads);
        }
        RandomCutForest forest = forestBuilder.build();
        RandomCutForest anotherForest = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).compact(true).boundingBoxCacheFraction(1.0).build();

        int count = 0;
        for (double[] point : generate(numTrainSamples, numDims, 0)) {
            ++count;
            double score = forest.getAnomalyScore(point);
            double anotherScore = anotherForest.getAnomalyScore(point);
            assertEquals(score, anotherScore, 1E-10);
            forest.update(point);
            anotherForest.update(point);
            if (count % 2000 == 1000) {
                double fraction = Math.random();
                // System.out.println(" second forest fraction " + fraction);
                anotherForest.setBoundingBoxCacheFraction(fraction);
            }
            if (count % 2000 == 0) {
                double fraction = Math.random();
                // System.out.println(" first forest fraction " + fraction);
                forest.setBoundingBoxCacheFraction(fraction);
            }
        }

    }

    @ParameterizedTest(name = "{index} => numDims={0}, numTrees={1}, numSamples={2}, numTrainSamples={3}, "
            + "numTestSamples={4}, enableParallel={5}, numThreads={6}")
    @CsvSource({ "10, 10, 30000, 50000, 0, 0, 0" })
    public void dynamicCachingChangeTestLarge(int numDims, int numTrees, int numSamples, int numTrainSamples,
            int numTestSamples, int enableParallel, int numThreads) {
        RandomCutForest.Builder<?> forestBuilder = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).boundingBoxCacheFraction(1.0).compact(false);
        if (enableParallel == 0) {
            forestBuilder.parallelExecutionEnabled(false);
        }
        if (numThreads > 0) {
            forestBuilder.threadPoolSize(numThreads);
        }
        RandomCutForest forest = forestBuilder.build();
        RandomCutForest anotherForest = RandomCutForest.builder().dimensions(numDims).numberOfTrees(numTrees)
                .sampleSize(numSamples).randomSeed(0).compact(true).boundingBoxCacheFraction(1.0).build();

        int count = 0;
        for (double[] point : generate(numTrainSamples, numDims, 0)) {
            ++count;
            double score = forest.getAnomalyScore(point);
            double anotherScore = anotherForest.getAnomalyScore(point);
            assertEquals(score, anotherScore, 1E-10);
            forest.update(point);
            anotherForest.update(point);
        }

    }

    private double[][] generate(int numSamples, int numDimensions, int seed) {
        return IntStream.range(0, numSamples).mapToObj(i -> new Random(seed + i).doubles(numDimensions).toArray())
                .toArray(double[][]::new);
    }
}
