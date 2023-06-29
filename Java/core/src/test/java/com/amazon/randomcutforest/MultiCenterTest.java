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
import static java.lang.Math.min;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.MultiCenter;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

@Tag("functional")
public class MultiCenterTest {

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    @Test
    public void constructorTest() {
        assertThrows(IllegalArgumentException.class, () -> MultiCenter.initialize(new float[4], 0, -1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> MultiCenter.initialize(new float[4], 0, 2.0, 1));
        assertThrows(IllegalArgumentException.class, () -> MultiCenter.initialize(new float[4], 0, 1.0, -1));
        assertThrows(IllegalArgumentException.class, () -> MultiCenter.initialize(new float[4], 0, 1.0, 1000));
    }

    @ParameterizedTest
    @MethodSource("generateArguments")
    public void SummaryTest(BiFunction<float[], float[], Double> distance) {

        int over = 0;
        int under = 0;

        for (int numTrials = 0; numTrials < 10; numTrials++) {
            long seed = new Random().nextLong();
            Random random = new Random(seed);
            int newDimensions = random.nextInt(10) + 3;
            dataSize = 200000;

            float[][] points = getData(dataSize, newDimensions, random.nextInt(), distance);

            List<ICluster<float[]>> summary = Summarizer.multiSummarize(points, 5 * newDimensions, 10 * newDimensions,
                    1, false, 0.8, distance, random.nextInt(), false, random.nextDouble(), 1);
            System.out.println("trial " + numTrials + " : " + summary.size() + " clusters for " + newDimensions
                    + " dimensions, seed : " + seed);
            if (summary.size() < 2 * newDimensions) {
                ++under;
            } else if (summary.size() > 2 * newDimensions) {
                ++over;
            }
        }
        assert (under <= 1);
    }

    @ParameterizedTest
    @MethodSource("generateArguments")
    public void MultiSummaryTestGeneric(BiFunction<float[], float[], Double> distance) {

        int over = 0;
        int under = 0;

        for (int numTrials = 0; numTrials < 10; numTrials++) {
            long seed = new Random().nextLong();
            Random random = new Random(seed);
            int newDimensions = random.nextInt(10) + 3;
            dataSize = 200000;

            float[][] points = getData(dataSize, newDimensions, random.nextInt(), distance);

            List<ICluster<float[]>> summary = Summarizer.multiSummarize(points, 5 * newDimensions, 10 * newDimensions,
                    1, false, 0.8, distance, random.nextInt(), false, random.nextDouble(), 5);
            System.out.println("trial " + numTrials + " : " + summary.size() + " clusters for " + newDimensions
                    + " dimensions, seed : " + seed);
            if (summary.size() < 2 * newDimensions) {
                ++under;
            } else if (summary.size() > 2 * newDimensions) {
                ++over;
            }
        }
        assert (under <= 1);
    }

    @Test
    public void MultiSummaryTest() {

        int over = 0;
        int under = 0;

        for (int numTrials = 0; numTrials < 10; numTrials++) {
            long seed = new Random().nextLong();
            Random random = new Random(seed);
            int newDimensions = random.nextInt(10) + 3;
            dataSize = 200000;

            float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);

            List<ICluster<float[]>> summary = Summarizer.multiSummarize(points, 5 * newDimensions, 0.9, 1, seed);
            System.out.println("trial " + numTrials + " : " + summary.size() + " clusters for " + newDimensions
                    + " dimensions, seed : " + seed);
            if (summary.size() < 2 * newDimensions) {
                ++under;
            } else if (summary.size() > 2 * newDimensions) {
                ++over;
            }
        }
        assert (under <= 1);
    }

    @ParameterizedTest
    @MethodSource("generateArguments")
    public void ParallelTest(BiFunction<float[], float[], Double> distance) {

        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), distance);
        System.out.println("checking parallelEnabled seed : " + seed);
        int nextSeed = random.nextInt();
        // these can differ for shinkage != 0 due to floating point issues
        List<ICluster<float[]>> summary1 = Summarizer.multiSummarize(points, 5 * newDimensions, 10 * newDimensions, 1,
                false, 0.8, distance, nextSeed, false, 0, 5);
        ArrayList<float[]> list = new ArrayList<>();
        for (float[] point : points) {
            list.add(point);
        }
        List<ICluster<float[]>> summary2 = Summarizer.multiSummarize(list, 5 * newDimensions, 10 * newDimensions, 1,
                false, 0.8, distance, nextSeed, true, 0, 5);

        assertEquals(summary2.size(), summary1.size(), " incorrect number of clusters");
        for (int i = 0; i < summary2.size(); i++) {
            assertEquals(summary1.get(i).getWeight(), summary2.get(i).getWeight(), 1e-6);
            assertEquals(summary1.get(i).extentMeasure(), summary2.get(i).extentMeasure(), 1e-6);
            List<Weighted<float[]>> reps1 = summary1.get(i).getRepresentatives();
            List<Weighted<float[]>> reps2 = summary2.get(i).getRepresentatives();
            assertEquals(reps1.size(), reps2.size());
            for (int j = 0; j < reps1.size(); j++) {
                assertEquals(reps1.get(j).weight, reps2.get(j).weight, 1e-6);
                assertArrayEquals(reps1.get(j).index, reps2.get(j).index, 1e-6f);
            }
        }

    }

    @Test
    public void StringTest() {

        long seed = new Random().nextLong();
        System.out.println("checking String summarization seed : " + seed);
        Random random = new Random(seed);
        int size = 100;
        int numberOfStrings = 20000;

        String[] points = new String[numberOfStrings];
        for (int i = 0; i < numberOfStrings; i++) {
            if (random.nextDouble() < 0.5) {
                points[i] = getABString(size, 0.8, random);
            } else {
                points[i] = getABString(size, 0.2, random);
            }
        }

        int nextSeed = random.nextInt();

        List<ICluster<String>> summary = Summarizer.multiSummarize(points, 5, 10, 1, false, 0.8,
                MultiCenterTest::toyDistance, nextSeed, false, 0.1, 5);
        System.out.println();
        assertEquals(summary.size(), 2);
    }

    public static double toyDistance(String a, String b) {
        if (a.length() > b.length()) {
            return toyDistance(b, a);
        }
        double[][] dist = new double[2][b.length() + 1];
        for (int j = 0; j < b.length() + 1; j++) {
            dist[0][j] = j;
        }

        for (int i = 1; i < a.length() + 1; i++) {
            dist[1][0] = i;
            for (int j = 1; j < b.length() + 1; j++) {
                double t = dist[0][j - 1] + ((a.charAt(i - 1) == b.charAt(j - 1)) ? 0 : 1);
                dist[1][j] = min(min(t, dist[0][j] + 1), dist[1][j - 1] + 1);
            }
            for (int j = 0; j < b.length() + 1; j++) {
                dist[0][j] = dist[1][j];
            }
        }
        return dist[1][b.length()];
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

    public String getABString(int size, double probabilityOfA, Random random) {
        StringBuilder stringBuilder = new StringBuilder();
        int newSize = size + random.nextInt(size / 5);
        for (int i = 0; i < newSize; i++) {
            if (random.nextDouble() < probabilityOfA) {
                stringBuilder.append("-");
            } else {
                stringBuilder.append("_");
            }
        }
        return stringBuilder.toString();
    }

    private static Stream<Arguments> generateArguments() {
        return Stream.of(Arguments.of((BiFunction<float[], float[], Double>) Summarizer::L1distance),
                Arguments.of((BiFunction<float[], float[], Double>) Summarizer::L2distance));
    }

}
