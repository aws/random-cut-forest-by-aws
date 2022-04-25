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

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;

import java.util.Random;
import java.util.function.BiFunction;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.imputation.Summarizer;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

@Tag("functional")
public class SampleSummaryTest {

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    @Test
    public void SummaryTest() {

        int over = 0;
        int under = 0;

        for (int numTrials = 0; numTrials < 100; numTrials++) {
            long seed = new Random().nextLong();
            Random random = new Random(seed);
            int newDimensions = random.nextInt(10) + 3;
            dataSize = 200000;

            float[][] points = getData(dataSize, newDimensions, random.nextInt());

            BiFunction<float[], float[], Double> L1distance = (a, b) -> {
                double dist = 0;
                for (int i = 0; i < a.length; i++) {
                    dist += Math.abs(a[i] - b[i]);
                }
                return dist;
            };

            BiFunction<float[], float[], Double> L2distance = (a, b) -> {
                double dist = 0;
                for (int i = 0; i < a.length; i++) {
                    double t = Math.abs(a[i] - b[i]);
                    dist += t * t;
                }
                return Math.sqrt(dist);
            };

            SampleSummary summary = Summarizer.summarize(points, 5 * newDimensions, 10 * newDimensions, false,
                    L2distance, random.nextInt());
            System.out.println("trial " + numTrials + " : " + summary.summaryPoints.length + " clusters for "
                    + newDimensions + " dimensions, seed : " + seed);
            if (summary.summaryPoints.length < 2 * newDimensions) {
                ++under;
            } else if (summary.summaryPoints.length > 2 * newDimensions) {
                ++over;
            }
        }
        assert (under <= 1);
        assert (over <= 1);
    }

    public float[][] getData(int dataSize, int newDimensions, int seed) {
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
        double[][] data = generator.generateTestData(dataSize, newDimensions, seed);
        float[][] floatData = new float[dataSize][];

        for (int i = 0; i < dataSize; i++) {
            // shrink, shift at random
            int nextD = prg.nextInt(newDimensions);
            for (int j = 0; j < newDimensions; j++) {
                data[i][j] *= 1.0 / (3.0 * Math.sqrt(newDimensions));
                // standard deviation adds up across dimension; taking square root
                // and using s 3 sigma ball
                if (j == nextD) {
                    if (prg.nextDouble() < 0.5)
                        data[i][j] += 2.0;
                    // set to 2*Math.sqrt(newDimensions); for L1
                    else
                        data[i][j] -= 2.0;
                    // set to 2*Math.sqrt(newDimensions); for L1
                }
            }
            floatData[i] = toFloatArray(data[i]);
        }

        return floatData;
    }

}
