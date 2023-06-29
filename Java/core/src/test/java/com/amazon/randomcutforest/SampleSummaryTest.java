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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

@Tag("functional")
public class SampleSummaryTest {

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    @ParameterizedTest
    @MethodSource("generateArguments")
    public void SummaryTest(BiFunction<float[], float[], Double> distance) {

        int over = 0;
        int under = 0;

        for (int numTrials = 0; numTrials < 20; numTrials++) {
            long seed = new Random().nextLong();
            Random random = new Random(seed);
            int newDimensions = random.nextInt(10) + 3;
            dataSize = 200000;

            float[][] points = getData(dataSize, newDimensions, random.nextInt(), distance);

            SampleSummary summary = Summarizer.summarize(points, 5 * newDimensions, 10 * newDimensions, false, distance,
                    random.nextInt(), false);
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

    @ParameterizedTest
    @MethodSource("generateArguments")
    public void ParallelTest(BiFunction<float[], float[], Double> distance) {

        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), distance);
        System.out.println("checking seed : " + seed);
        int nextSeed = random.nextInt();
        SampleSummary summary1 = Summarizer.summarize(points, 5 * newDimensions, 10 * newDimensions, false, distance,
                nextSeed, false);
        SampleSummary summary2 = Summarizer.summarize(points, 5 * newDimensions, 10 * newDimensions, false, distance,
                nextSeed, true);

        ArrayList<Weighted<float[]>> pointList = new ArrayList<>();
        for (float[] point : points) {
            pointList.add(new Weighted<>(point, 1.0f));
        }
        List<ICluster<float[]>> clusters = Summarizer.singleCentroidSummarize(pointList, 5 * newDimensions,
                10 * newDimensions, 1, true, distance, nextSeed, false, null);
        assertEquals(summary2.weightOfSamples, summary1.weightOfSamples, " sampling inconsistent");
        assertEquals(summary2.summaryPoints.length, summary1.summaryPoints.length,
                " incorrect length of typical points");
        assertEquals(clusters.size(), summary1.summaryPoints.length);
        double total = clusters.stream().map(ICluster::getWeight).reduce(0.0, Double::sum);
        assertEquals(total, summary1.weightOfSamples, 1e-3);
        // parallelization can produce reordering of merges
    }

    @Test
    public void SampleSummaryTestL2() {
        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);
        System.out.println("checking L2 seed : " + seed);
        int nextSeed = random.nextInt();
        ArrayList<Weighted<float[]>> pointList = new ArrayList<>();
        for (float[] point : points) {
            pointList.add(new Weighted<>(point, 1.0f));
        }
        SampleSummary summary1 = Summarizer.summarize(points, 5 * newDimensions, 20 * newDimensions, false,
                Summarizer::L2distance, nextSeed, false);
        SampleSummary summary2 = Summarizer.l2summarize(points, 5 * newDimensions, nextSeed);
        SampleSummary summary3 = Summarizer.l2summarize(pointList, 5 * newDimensions, 20 * newDimensions, false,
                nextSeed);

        assertEquals(summary2.weightOfSamples, summary1.weightOfSamples, " sampling inconsistent");
        assertEquals(summary3.weightOfSamples, summary1.weightOfSamples, " sampling inconsistent");
        assertEquals(summary2.summaryPoints.length, summary1.summaryPoints.length,
                " incorrect length of typical points");
        assertEquals(summary3.summaryPoints.length, summary1.summaryPoints.length,
                " incorrect length of typical points");
        for (int i = 0; i < summary2.summaryPoints.length; i++) {
            assertArrayEquals(summary1.summaryPoints[i], summary2.summaryPoints[i], 1e-6f);
            assertArrayEquals(summary1.summaryPoints[i], summary3.summaryPoints[i], 1e-6f);
            assertEquals(summary1.relativeWeight[i], summary2.relativeWeight[i], 1e-6f);
            assertEquals(summary1.relativeWeight[i], summary3.relativeWeight[i], 1e-6f);
        }
    }

    @Test
    public void IdempotenceTestL2() {

        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);
        System.out.println("checking idempotence L2 seed : " + seed);
        int nextSeed = random.nextInt();
        ArrayList<Weighted<float[]>> pointList = new ArrayList<>();
        for (float[] point : points) {
            pointList.add(new Weighted<>(point, 1.0f));
        }
        List<ICluster<float[]>> clusters = Summarizer.singleCentroidSummarize(pointList, 5 * newDimensions,
                20 * newDimensions, 1, true, Summarizer::L2distance, nextSeed, false, null);
        List<ICluster<float[]>> clusters2 = Summarizer.singleCentroidSummarize(pointList, 5 * newDimensions,
                20 * newDimensions, 1, true, Summarizer::L2distance, nextSeed, false, clusters);
        assertEquals(clusters.size(), clusters2.size(), " incorrect sizes");
        for (int i = 0; i < clusters.size(); i++) {
            // note clusters can have same weight and get permuted
            assertEquals(clusters.get(i).getWeight(), clusters2.get(i).getWeight());
        }
        clusters.sort(Comparator.comparingDouble(ICluster::extentMeasure));
        clusters2.sort(Comparator.comparingDouble(ICluster::extentMeasure));
        assertEquals(clusters.size(), clusters2.size(), " incorrect sizes");
        for (int i = 0; i < clusters.size(); i++) {
            // note clusters can have same weight and get permuted
            assertEquals(clusters.get(i).extentMeasure(), clusters2.get(i).extentMeasure());
            assertEquals(clusters.get(i).averageRadius(), clusters2.get(i).averageRadius());
            assertEquals(clusters.get(i).averageRadius(), clusters.get(i).extentMeasure());
        }
    }

    public float[][] getData(int dataSize, int newDimensions, int seed, BiFunction<float[], float[], Double> distance) {
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

        float[] allZero = new float[newDimensions];
        float[] sigma = new float[newDimensions];
        Arrays.fill(sigma, 1f);
        double scale = distance.apply(allZero, sigma);

        for (int i = 0; i < dataSize; i++) {
            // shrink, shift at random
            int nextD = prg.nextInt(newDimensions);
            for (int j = 0; j < newDimensions; j++) {
                data[i][j] *= 1.0 / (3.0);
                // standard deviation adds up across dimension; taking square root
                // and using s 3 sigma ball
                if (j == nextD) {
                    if (prg.nextDouble() < 0.5)
                        data[i][j] += 2.0 * scale;
                    else
                        data[i][j] -= 2.0 * scale;
                }
            }
            floatData[i] = toFloatArray(data[i]);
        }

        return floatData;
    }

    private static Stream<Arguments> generateArguments() {
        return Stream.of(Arguments.of((BiFunction<float[], float[], Double>) Summarizer::L1distance),
                Arguments.of((BiFunction<float[], float[], Double>) Summarizer::L2distance),
                Arguments.of((BiFunction<float[], float[], Double>) Summarizer::LInfinitydistance));
    }

}
