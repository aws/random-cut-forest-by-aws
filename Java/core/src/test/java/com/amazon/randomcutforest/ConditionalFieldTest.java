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

import java.util.Random;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class ConditionalFieldTest {

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
    public void SimpleTest() {

        int newDimensions = 30;
        randomSeed = 101;
        sampleSize = 256;
        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(sampleSize)
                .dimensions(newDimensions).randomSeed(randomSeed).compact(true).boundingBoxCacheFraction(0.0).build();

        dataSize = 2000 + 5;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 0.0;
        anomalySigma = 1.0;
        transitionToAnomalyProbability = 0.0;
        // ignoring anomaly cluster for now
        transitionToBaseProbability = 1.0;
        Random prg = new Random(0);
        NormalMixtureTestData generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, newDimensions, 100);

        for (int i = 0; i < 2000; i++) {
            // shrink, shift at random
            for (int j = 0; j < newDimensions; j++)
                data[i][j] *= 0.01;
            if (prg.nextDouble() < 0.5)
                data[i][0] += 5.0;
            else
                data[i][0] -= 5.0;
            newForest.update(data[i]);
        }

        float[] queryOne = new float[newDimensions];
        float[] queryTwo = new float[newDimensions];
        queryTwo[1] = 1;
        SampleSummary summary = newForest.getConditionalFieldSummary(queryOne, 1, new int[] { 0 }, 1, 0, true, false,
                1);

        assert (summary.summaryPoints.length == 2);
        assert (summary.relativeWeight.length == 2);
        assert (Math.abs(summary.summaryPoints[0][0] - 5.0) < 0.01
                || Math.abs(summary.summaryPoints[0][0] + 5.0) < 0.01);
        assert (Math.abs(summary.summaryPoints[1][0] - 5.0) < 0.01
                || Math.abs(summary.summaryPoints[1][0] + 5.0) < 0.01);
        assert (summary.relativeWeight[0] > 0.25);
        assert (summary.relativeWeight[1] > 0.25);

        summary = newForest.getConditionalFieldSummary(queryTwo, 1, new int[] { 0 }, 1, 0, true, false, 1);

        assert (summary.summaryPoints.length == 2);
        assert (summary.relativeWeight.length == 2);
        assertEquals(summary.summaryPoints[0][1], 1, 1e-6);
        assertEquals(summary.summaryPoints[1][1], 1, 1e-6);
        assert (Math.abs(summary.summaryPoints[0][0] - 5.0) < 0.01
                || Math.abs(summary.summaryPoints[0][0] + 5.0) < 0.01);
        assert (Math.abs(summary.summaryPoints[1][0] - 5.0) < 0.01
                || Math.abs(summary.summaryPoints[1][0] + 5.0) < 0.01);
        assert (summary.relativeWeight[0] > 0.25);
        assert (summary.relativeWeight[1] > 0.25);

    }

}
