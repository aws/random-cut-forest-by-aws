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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("functional")
public class AttributionExamplesFunctionalTest {

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

    @Test
    public void RRCFattributionTest() {

        // starts with the same setup as rrcfTest; data corresponds to two small
        // clusters at x=+/-5.0
        // queries q_1=(0,0,0, ..., 0)
        // inserts updates (0,1,0, ..., 0) a few times
        // queries q_2=(0,1,0, ..., 0)
        // attribution of q_2 is now affected by q_1 (which is still an anomaly)

        int newDimensions = 30;
        randomSeed = 101;
        sampleSize = 256;
        RandomCutForest newForest =
                RandomCutForest.builder()
                        .numberOfTrees(100)
                        .sampleSize(sampleSize)
                        .dimensions(newDimensions)
                        .randomSeed(randomSeed)
                        .compact(false)
                        .boundingBoxCacheFraction(0.0)
                        .build();

        dataSize = 2000 + 5;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 0.0;
        anomalySigma = 1.0;
        transitionToAnomalyProbability = 0.0;
        // ignoring anomaly cluster for now
        transitionToBaseProbability = 1.0;
        Random prg = new Random(0);
        NormalMixtureTestData generator =
                new NormalMixtureTestData(
                        baseMu,
                        baseSigma,
                        anomalyMu,
                        anomalySigma,
                        transitionToAnomalyProbability,
                        transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, newDimensions, 100);

        for (int i = 0; i < 2000; i++) {
            // shrink, shift at random
            for (int j = 0; j < newDimensions; j++) data[i][j] *= 0.01;
            if (prg.nextDouble() < 0.5) data[i][0] += 5.0;
            else data[i][0] -= 5.0;
            newForest.update(data[i]);
        }

        double[] queryOne = new double[newDimensions];
        double[] queryTwo = new double[newDimensions];
        queryTwo[1] = 1;
        double originalScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector originalAttrTwo = newForest.getAnomalyAttribution(queryTwo);

        assertTrue(originalScoreTwo > 3.0);
        assertEquals(originalScoreTwo, originalAttrTwo.getHighLowSum(), 1E-10);

        assertTrue(originalAttrTwo.high[0] > 1.0); // due to -5 cluster
        assertTrue(originalAttrTwo.low[0] > 1.0); // due to +5 cluster
        assertTrue(originalAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(originalAttrTwo.getHighLowSum(0) > 1.1 * originalAttrTwo.getHighLowSum(1));

        // we insert queryOne a few times to make sure it is sampled
        for (int i = 2000; i < 2000 + 5; i++) {
            double score = newForest.getAnomalyScore(queryOne);
            double score2 = newForest.getAnomalyScore(queryTwo);
            DiVector attr2 = newForest.getAnomalyAttribution(queryTwo);

            // verify
            assertTrue(score > 2.0);
            assertTrue(score2 > 2.0);
            assertEquals(attr2.getHighLowSum(), score2, 1E-10);

            for (int j = 0; j < newDimensions; j++) data[i][j] *= 0.01;
            newForest.update(data[i]);
            // 5 different anomalous points
        }

        double midScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector midAttrTwo = newForest.getAnomalyAttribution(queryTwo);

        assertTrue(midScoreTwo > 2.4);
        assertEquals(midScoreTwo, midAttrTwo.getHighLowSum(), 1E-10);

        assertTrue(midAttrTwo.high[0] < 1); // due to -5 cluster !!!
        assertTrue(midAttrTwo.low[0] < 1); // due to +5 cluster !!!
        assertTrue(midAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(midAttrTwo.getHighLowSum(0) < 1.1 * midAttrTwo.high[1]);
        // reversal of the dominant dimension
        // still an anomaly; but the attribution is masked by points

        // a few more updates, which are identical
        for (int i = 2005; i < 2010; i++) {
            newForest.update(queryOne);
        }

        double finalScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector finalAttrTwo = newForest.getAnomalyAttribution(queryTwo);
        assertTrue(finalScoreTwo > 2.4);
        assertEquals(finalScoreTwo, finalAttrTwo.getHighLowSum(), 1E-10);
        assertTrue(finalAttrTwo.high[0] < 0.5); // due to -5 cluster !!!
        assertTrue(finalAttrTwo.low[0] < 0.5); // due to +5 cluster !!!
        assertTrue(finalAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(2.5 * finalAttrTwo.getHighLowSum(0) < finalAttrTwo.high[1]);
        // the drop in high[0] and low[0] is steep and the attribution has shifted

    }

    @Test
    public void attributionUnMaskingTest() {

        // starts with the same setup as rrcfTest; data corresponds to two small
        // clusters at x=+/-5.0
        // queries q_1=(0,0,0, ..., 0)
        // inserts updates (0,1,0, ..., 0) a few times
        // queries q_2=(0,1,0, ..., 0)
        // attribution of q_2 is now affected by q_1 (which is still an anomaly)

        int newDimensions = 30;
        randomSeed = 179;
        sampleSize = 256;
        DynamicScoringRandomCutForest newForest =
                DynamicScoringRandomCutForest.builder()
                        .numberOfTrees(100)
                        .sampleSize(sampleSize)
                        .dimensions(newDimensions)
                        .randomSeed(randomSeed)
                        .compact(false)
                        .boundingBoxCacheFraction(1.0)
                        .lambda(1e-5)
                        .build();

        dataSize = 2000 + 5;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 0.0;
        anomalySigma = 1.5;
        transitionToAnomalyProbability = 0.0;
        // ignoring anomaly cluster for now
        transitionToBaseProbability = 1.0;
        Random prg = new Random(0);
        NormalMixtureTestData generator =
                new NormalMixtureTestData(
                        baseMu,
                        baseSigma,
                        anomalyMu,
                        anomalySigma,
                        transitionToAnomalyProbability,
                        transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, newDimensions, 100);

        for (int i = 0; i < 2000; i++) {
            // shrink, shift at random
            for (int j = 0; j < newDimensions; j++) data[i][j] *= 0.01;
            if (prg.nextDouble() < 0.5) data[i][0] += 5.0;
            else data[i][0] -= 5.0;
            newForest.update(data[i]);
        }

        double[] queryOne = new double[30];
        double[] queryTwo = new double[30];
        queryTwo[1] = 1;
        double originalScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector originalAttrTwo =
                newForest.getDynamicAttribution(
                        queryTwo,
                        0,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);

        originalAttrTwo.componentwiseTransform(
                x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));
        assertTrue(originalScoreTwo > 3.0);

        assertEquals(originalScoreTwo, originalAttrTwo.getHighLowSum(), 1E-10);

        assertTrue(originalAttrTwo.high[0] > 0.75); // due to -5 cluster
        assertTrue(originalAttrTwo.low[0] > 0.75); // due to +5 cluster
        assertTrue(originalAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(originalAttrTwo.getHighLowSum(0) > originalAttrTwo.getHighLowSum(1));

        // we insert queryOne a few times to make sure it is sampled
        for (int i = 2000; i < 2000 + 5; i++) {
            double score = newForest.getAnomalyScore(queryOne);
            double score2 = newForest.getAnomalyScore(queryTwo);
            DiVector attr2 =
                    newForest.getDynamicAttribution(
                            queryTwo,
                            0,
                            CommonUtils::defaultScoreSeenFunction,
                            CommonUtils::defaultScoreUnseenFunction,
                            CommonUtils::defaultDampFunction);
            attr2.componentwiseTransform(
                    x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));

            double score3 =
                    newForest.getDynamicScore(
                            queryTwo,
                            1,
                            CommonUtils::defaultScoreSeenFunction,
                            CommonUtils::defaultScoreUnseenFunction,
                            CommonUtils::defaultDampFunction);
            score3 = CommonUtils.defaultScalarNormalizerFunction(score3, sampleSize);
            DiVector attr3 =
                    newForest.getDynamicAttribution(
                            queryTwo,
                            1,
                            CommonUtils::defaultScoreSeenFunction,
                            CommonUtils::defaultScoreUnseenFunction,
                            CommonUtils::defaultDampFunction);
            attr3.componentwiseTransform(
                    x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));

            // verify
            assertTrue(score > 2.0);
            assertTrue(score2 > 2.0);
            assertTrue(score3 > 2.0);
            assertEquals(attr2.getHighLowSum(), score2, 1E-10);
            assertEquals(attr3.getHighLowSum(), score3, 1E-10);

            for (int j = 0; j < newDimensions; j++) data[i][j] *= 0.01;
            newForest.update(data[i]);
            // 5 different anomalous points
        }

        double midScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector midAttrTwo =
                newForest.getDynamicAttribution(
                        queryTwo,
                        0,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        midAttrTwo.componentwiseTransform(
                x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));

        assertTrue(midScoreTwo > 2.5);
        assertEquals(midScoreTwo, midAttrTwo.getHighLowSum(), 1E-10);

        assertTrue(midAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(midAttrTwo.getHighLowSum(0) < 1.2 * midAttrTwo.high[1]);
        // reversal of the dominant dimension
        // still an anomaly; but the attribution is masked by points

        double midUnmaskedScore =
                newForest.getDynamicScore(
                        queryTwo,
                        1,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        midUnmaskedScore =
                CommonUtils.defaultScalarNormalizerFunction(midUnmaskedScore, sampleSize);
        DiVector midUnmaskedAttr =
                newForest.getDynamicAttribution(
                        queryTwo,
                        1,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        midUnmaskedAttr.componentwiseTransform(
                x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));

        assertTrue(midUnmaskedScore > 3.0);
        assertEquals(midUnmaskedScore, midUnmaskedAttr.getHighLowSum(), 1E-10);

        assertTrue(midUnmaskedAttr.high[1] > 1); // due to +1 in query
        assertTrue(midUnmaskedAttr.getHighLowSum(0) > midUnmaskedAttr.getHighLowSum(1));
        // contribution from dimension 0 is still dominant
        // the attributions in dimension 0 are reduced, but do not
        // or become as small as quickly as in the other case

        // a few more updates, which are identical
        for (int i = 2005; i < 2010; i++) {
            newForest.update(queryOne);
        }

        double finalScoreTwo = newForest.getAnomalyScore(queryTwo);
        DiVector finalAttrTwo =
                newForest.getDynamicAttribution(
                        queryTwo,
                        0,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        finalAttrTwo.componentwiseTransform(
                x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));
        assertTrue(finalScoreTwo > 2.5);
        assertEquals(finalScoreTwo, finalAttrTwo.getHighLowSum(), 1E-10);

        assertTrue(finalAttrTwo.high[1] > 1); // due to +1 in query
        assertTrue(2 * finalAttrTwo.getHighLowSum(0) < finalAttrTwo.high[1]);
        // the drop in high[0] and low[0] is steep and the attribution has shifted

        // different thresholds
        double finalUnmaskedScore =
                newForest.getDynamicScore(
                        queryTwo,
                        5,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        finalUnmaskedScore =
                CommonUtils.defaultScalarNormalizerFunction(finalUnmaskedScore, sampleSize);
        DiVector finalUnmaskedAttr =
                newForest.getDynamicAttribution(
                        queryTwo,
                        5,
                        CommonUtils::defaultScoreSeenFunction,
                        CommonUtils::defaultScoreUnseenFunction,
                        CommonUtils::defaultDampFunction);
        finalUnmaskedAttr.componentwiseTransform(
                x -> CommonUtils.defaultScalarNormalizerFunction(x, sampleSize));

        assertTrue(finalUnmaskedScore > 3.0);
        assertEquals(finalUnmaskedScore, finalUnmaskedAttr.getHighLowSum(), 1E-10);

        assertTrue(finalUnmaskedAttr.high[1] > 1); // due to +1 in query
        assertTrue(finalUnmaskedAttr.getHighLowSum(0) > 0.8 * finalUnmaskedAttr.getHighLowSum(1));

        // the attributions in dimension 0 continue to be reduced, but do not vanish
        // or become small as in the other case; the gap is not a factor of 4
    }
}
