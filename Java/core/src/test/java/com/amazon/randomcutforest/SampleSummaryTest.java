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
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.Center;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.MultiCenter;
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

    @Test
    public void configAndAbsorbTest() {

        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        dataSize = 2000;
        Summarizer summarizer = new Summarizer();

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);
        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        ArrayList<Weighted<Integer>> refs = new ArrayList<>();

        int count = 0;
        for (float[] point : points) {
            // testing 0 weight
            weighted.add(new Weighted<>(point, 0.0f));
            refs.add(new Weighted<Integer>(count, 0.0f));
            ++count;
        }

        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 500, 10 * newDimensions,
                false, Summarizer::L2distance, random.nextInt(), false));
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> MultiCenter.initialize(a, b, 0.8,
                3);
        Function<Integer, float[]> getPoint = (i) -> weighted.get(i).index;

        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 500, 10 * newDimensions, 1,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> Summarizer.summarize(weighted, 50, 10, false, Summarizer::L2distance, random.nextInt(), false));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 50, 10, 1, false, 0.1,
                Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 50, 10 * newDimensions, 0,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 50, 10 * newDimensions, 100,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 0,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 7,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        assertThrows(IllegalArgumentException.class,
                () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1, Collections.emptyList(), getPoint,
                        Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, false,
                Summarizer::L2distance, random.nextInt(), false));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, 1,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        Weighted<float[]> a = weighted.get(0);
        a.weight = -1;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, 1,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, false,
                Summarizer::L2distance, random.nextInt(), false));
        a.weight = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, 1,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, false,
                Summarizer::L2distance, random.nextInt(), false));
        a.weight = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, 1,
                false, 0.1, Summarizer::L2distance, clusterInitializer, 0, false, null));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.summarize(weighted, 5, 10 * newDimensions, false,
                Summarizer::L2distance, random.nextInt(), false));
        a.weight = 1;
        assertDoesNotThrow(() -> Summarizer.summarize(weighted, 5, 10 * newDimensions, false, Summarizer::L2distance,
                random.nextInt(), false));
        assertDoesNotThrow(() -> Summarizer.summarize(weighted, 5, 10 * newDimensions, 1, false, 0.1,
                Summarizer::L2distance, clusterInitializer, 0, false, null));

        refs.get(0).weight = -1;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        refs.get(0).weight = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        refs.get(0).weight = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        refs.get(0).weight = 0;
        assertThrows(IllegalArgumentException.class, () -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1,
                refs, getPoint, Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));
        refs.get(0).weight = 1;
        assertDoesNotThrow(() -> Summarizer.iterativeClustering(5, 10 * newDimensions, 1, refs, getPoint,
                Summarizer::L2distance, clusterInitializer, 0, false, false, 0.1, null));

        assertThrows(IllegalArgumentException.class, () -> Summarizer.assignAndRecompute(refs, getPoint,
                Collections.emptyList(), Summarizer::L2distance, false));
        List<ICluster<float[]>> list = new ArrayList<>();
        list.add(clusterInitializer.apply(new float[newDimensions], 1f));
        assertThrows(IllegalArgumentException.class, () -> Summarizer.assignAndRecompute(Collections.emptyList(),
                getPoint, list, Summarizer::L2distance, false));
        assertDoesNotThrow(() -> Summarizer.assignAndRecompute(refs, getPoint, list, Summarizer::L2distance, false));
        assertArrayEquals(list.get(0).primaryRepresentative(Summarizer::L2distance), new float[newDimensions], 1e-6f);

        float[] newPoint = new float[newDimensions];
        Arrays.fill(newPoint, 1.01f);
        list.get(0).absorb(clusterInitializer.apply(newPoint, 1f), Summarizer::L2distance);
        BiFunction<float[], float[], Double> badDistance = mock();
        when(badDistance.apply(any(), any())).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class,
                () -> Summarizer.assignAndRecompute(refs, getPoint, list, badDistance, false));
    }

    @Test
    public void TestMultiCenter() {
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> MultiCenter.initialize(a, b, 0.8,
                3);
        Function<Integer, float[]> getPoint = (i) -> {
            return new float[1];
        };
        ICluster<float[]> newCluster = clusterInitializer.apply(new float[1], 1f);
        float[] newPoint = new float[] { 1 };
        BiFunction<float[], float[], Double> badDistance = mock();
        when(badDistance.apply(any(), any())).thenReturn(-1.0);
        ICluster<float[]> cluster = clusterInitializer.apply(new float[1], 1.0f);
        ICluster<float[]> another = clusterInitializer.apply(new float[1], 1.0f);
        assertThrows(IllegalArgumentException.class, () -> cluster.absorb(another, badDistance));
        when(badDistance.apply(any(), any())).thenReturn(-1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> cluster.distance(new float[1], badDistance));
        assertThrows(IllegalArgumentException.class, () -> cluster.absorb(another, badDistance));

        newCluster.absorb(clusterInitializer.apply(newPoint, 1f), Summarizer::L2distance);
        when(badDistance.apply(any(), any())).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster.absorb(another, badDistance));

        ICluster<float[]> newCluster2 = clusterInitializer.apply(new float[1], 1f);
        newCluster2.absorb(clusterInitializer.apply(newPoint, 1f), Summarizer::L2distance);
        when(badDistance.apply(any(), any())).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0)
                .thenReturn(1.0);
        newCluster2.absorb(clusterInitializer.apply(newPoint, 1f), badDistance);
        when(badDistance.apply(any(), any())).thenReturn(1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster2.distance(new float[1], badDistance));
        another.absorb(clusterInitializer.apply(newPoint, 1f), Summarizer::L2distance);
        when(badDistance.apply(any(), any())).thenReturn(-1.0).thenReturn(1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster2.distance(another, badDistance));
        // error at a different location
        assertThrows(IllegalArgumentException.class, () -> newCluster2.distance(another, badDistance));
        when(badDistance.apply(any(), any())).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0)
                .thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0).thenReturn(1.0)
                .thenReturn(1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster2.absorb(another, badDistance));

        ICluster<float[]> newCluster3 = MultiCenter.initialize(new float[1], 0f, 0, 1);
        assertEquals(newCluster3.recompute(getPoint, false, Summarizer::L2distance), 0);
        assertEquals(newCluster3.recompute(getPoint, true, Summarizer::L2distance), 0);
        newCluster3.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        assertEquals(newCluster3.recompute(getPoint, true, Summarizer::L2distance), 0);

        ICluster<float[]> newCluster4 = MultiCenter.initialize(new float[1], 1f, 0, 1);
        when(badDistance.apply(any(), any())).thenReturn(-1.0).thenReturn(-1.0);
        newCluster4.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> newCluster4.recompute(getPoint, true, badDistance));
        assertThrows(IllegalArgumentException.class, () -> newCluster4.absorb(newCluster3, badDistance));

    }

    @Test
    public void testCenter() {
        int newDimensions = 1;
        Function<Integer, float[]> getPoint = (i) -> {
            return new float[1];
        };
        BiFunction<float[], float[], Double> badDistance = mock();
        ICluster<float[]> newCluster5 = Center.initialize(new float[newDimensions], 0f);
        assertEquals(newCluster5.extentMeasure(), newCluster5.averageRadius());
        assertEquals(newCluster5.recompute(getPoint, true, Summarizer::L2distance), 0);
        newCluster5.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        assertEquals(newCluster5.recompute(getPoint, true, Summarizer::L2distance), 0);
        when(badDistance.apply(any(), any())).thenReturn(-1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster5.distance(new float[1], badDistance));

        ICluster<float[]> newCluster6 = Center.initialize(new float[newDimensions], 10f);
        newCluster6.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        newCluster6.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        when(badDistance.apply(any(), any())).thenReturn(-1.0).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster6.absorb(newCluster5, badDistance));
        assertThrows(IllegalArgumentException.class, () -> newCluster6.recompute(getPoint, true, badDistance));
        ICluster<float[]> multiCenter1 = MultiCenter.initialize(new float[] { 1 }, 5.0f, 0.8, 2);
        ICluster<float[]> multiCenter2 = MultiCenter.initialize(new float[] { 2 }, 5.0f, 0.8, 2);
        multiCenter1.absorb(multiCenter2, Summarizer::L2distance); // weight 10
        newCluster6.absorb(multiCenter1, Summarizer::L2distance);
        assertEquals(newCluster6.primaryRepresentative(Summarizer::L2distance)[0], 0.5, 1e-6f);

        ICluster<float[]> newCluster7 = Center.initialize(new float[newDimensions], -10f);
        newCluster7.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        newCluster7.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        when(badDistance.apply(any(), any())).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster7.recompute(getPoint, true, badDistance));

        ICluster<float[]> newCluster8 = Center.initialize(new float[newDimensions], 1.9f);
        newCluster8.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        newCluster8.getAssignedPoints().add(new Weighted<>(1, 1.0f));
        when(badDistance.apply(any(), any())).thenReturn(-1.0);
        assertThrows(IllegalArgumentException.class, () -> newCluster8.recompute(getPoint, true, badDistance));
    }

    @Test
    public void zeroTest() {
        Random random = new Random(0);
        dataSize = 2000;

        float[][] points = new float[dataSize][];
        for (int y = 0; y < dataSize; y++) {
            points[y] = new float[] { (float) (random.nextInt(100) + 0.5 * random.nextDouble()) };
        }

        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        Function<Integer, float[]> getPoint = (x) -> weighted.get(x).index;
        int count = 0;
        for (float[] point : points) {
            // testing 0 weight
            weighted.add(new Weighted<>(point, 1.0f));
            refs.add(new Weighted<Integer>(count, 1.0f));
            ++count;
        }
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> Center.initialize(a, b);
        List<ICluster<float[]>> list = new ArrayList<>();
        for (int y = 0; y < 200; y++) {
            list.add(clusterInitializer.apply(new float[] { -1.0f }, 1.0f));
        }
        assertDoesNotThrow(() -> Summarizer.iterativeClustering(100, 0, 1, refs, getPoint, Summarizer::L2distance,
                clusterInitializer, 0, false, true, 0.1, list));
    }

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
