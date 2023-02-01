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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.testutils.ExampleDataSets.rotateClockWise;
import static java.lang.Math.PI;
import static java.lang.Math.min;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.function.BiFunction;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.parkservices.returntypes.GenericAnomalyDescriptor;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

public class TestGlobalLocalAnomalyDetector {

    @Test
    void testDynamicStringClustering() {
        long seed = new Random().nextLong();
        System.out.println("String summarization seed : " + seed);
        Random random = new Random(seed);
        int stringSize = 70;
        int numberOfStrings = 200000;
        int reservoirSize = 2000;
        boolean changeInMiddle = true;
        // the following should be away from 0.5 in [0.5,1]
        double gapProbOfA = 0.85;

        double anomalyRate = 0.05;
        char[][] points = new char[numberOfStrings][];
        boolean[] injected = new boolean[numberOfStrings];
        int numberOfInjected = 0;

        for (int i = 0; i < numberOfStrings; i++) {
            if (random.nextDouble() < anomalyRate && i > reservoirSize / 2) {
                injected[i] = true;
                ++numberOfInjected;
                points[i] = getABArray(stringSize + 10, 0.5, random, false, 0);
            } else {
                boolean flag = changeInMiddle && random.nextDouble() < 0.25;
                double prob = (random.nextDouble() < 0.5) ? gapProbOfA : (1 - gapProbOfA);
                points[i] = getABArray(stringSize, prob, random, flag, 0.25 * i / numberOfStrings);
            }
        }

        System.out.println("Injected " + numberOfInjected + " 'anomalies' in " + points.length);
        int recluster = reservoirSize / 2;

        BiFunction<char[], char[], Double> dist = (a, b) -> toyD(a, b, stringSize / 2.0);
        GlobalLocalAnomalyDetector<char[]> reservoir = GlobalLocalAnomalyDetector.builder().randomSeed(42)
                .numberOfRepresentatives(5).timeDecay(1.0 / reservoirSize).capacity(reservoirSize).build();
        reservoir.setGlobalDistance(dist);

        int truePos = 0;
        int falsePos = 0;
        int falseNeg = 0;
        for (int y = 0; y < points.length; y++) {

            if (y % 200 == 100 && y > reservoirSize) {
                char[] temp = points[y];
                // check for malformed distance function, to the extent we can check efficiently
                BiFunction<char[], char[], Double> badDistance = (a, b) -> -1.0;
                assertThrows(IllegalArgumentException.class, () -> {
                    reservoir.process(temp, 1.0f, badDistance, true);
                });
            }
            GenericAnomalyDescriptor<char[]> result = reservoir.process(points[y], 1.0f, null, true);

            if (result.getRepresentativeList() != null) {
                double sum = 0;
                for (Weighted<char[]> rep : result.getRepresentativeList()) {
                    assert (rep.weight <= 1.0);
                    sum += rep.weight;
                }
                // checking likelihood summing to 1
                assertEquals(sum, 1.0, 1e-6);
            }

            if (result.getAnomalyGrade() > 0) {
                if (!injected[y]) {
                    ++falsePos;
                } else {
                    ++truePos;
                }
            } else if (injected[y]) {
                ++falseNeg;
            }

            if (10 * y % points.length == 0 && y > 0) {
                System.out.println(" at " + y);
                System.out.println("Precision = " + precision(truePos, falsePos));
                System.out.println("Recall = " + recall(truePos, falseNeg));
            }
        }
        System.out.println(" Final: ");
        System.out.println("Precision = " + precision(truePos, falsePos));
        System.out.println("Recall = " + recall(truePos, falseNeg));
    }

    public static double toyD(char[] a, char[] b, double u) {
        if (a.length > b.length) {
            return toyD(b, a, u);
        }
        double[][] dist = new double[2][b.length + 1];
        for (int j = 0; j < b.length + 1; j++) {
            dist[0][j] = j;
        }

        for (int i = 1; i < a.length + 1; i++) {
            dist[1][0] = i;
            for (int j = 1; j < b.length + 1; j++) {
                double t = dist[0][j - 1] + ((a[i - 1] == b[j - 1]) ? 0 : 1);
                dist[1][j] = min(min(t, dist[0][j] + 1), dist[1][j - 1] + 1);
            }
            for (int j = 0; j < b.length + 1; j++) {
                dist[0][j] = dist[1][j];
            }
        }
        return dist[1][b.length];
    }

    // colors
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_BLUE = "\u001B[34m";

    public char[] getABArray(int size, double probabilityOfA, Random random, Boolean changeInMiddle, double fraction) {

        int newSize = size + random.nextInt(size / 5);
        char[] a = new char[newSize];
        for (int i = 0; i < newSize; i++) {
            double toss = (changeInMiddle && (i > (1 - fraction) * newSize || i < newSize * fraction))
                    ? (1 - probabilityOfA)
                    : probabilityOfA;
            if (random.nextDouble() < toss) {
                a[i] = '-';
            } else {
                a[i] = '_';
            }
        }
        return a;
    }

    public double[][] shiftedEllipse(int dataSize, int seed, double shift, int fans) {
        NormalMixtureTestData generator = new NormalMixtureTestData(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        double[][] data = generator.generateTestData(dataSize, 2, seed);
        Random prg = new Random(0);
        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale
            data[i][1] *= 1.0 / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += shift + 1.0 / fans;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    @Test
    void testDynamicNumericClustering() throws IOException {
        long randomSeed = new Random().nextLong();
        System.out.println("Seed " + randomSeed);
        // we would be sending dataSize * 360 vectors
        int dataSize = 2000;
        double range = 10.0;
        int numberOfFans = 3;
        // corresponds to number of clusters
        double[][] data = shiftedEllipse(dataSize, 7, range / 2, numberOfFans);
        int truePos = 0;
        int falsePos = 0;
        int falseNeg = 0;

        int truePosRCF = 0;
        int falsePosRCF = 0;
        int falseNegRCF = 0;

        int reservoirSize = dataSize;
        double timedecay = 1.0 / reservoirSize;
        GlobalLocalAnomalyDetector<float[]> reservoir = GlobalLocalAnomalyDetector.builder().randomSeed(42)
                .numberOfRepresentatives(3).timeDecay(timedecay).capacity(reservoirSize).build();
        reservoir.setGlobalDistance(Summarizer::L2distance);

        double zFactor = 6.0; // six sigma deviation; seems to work best
        reservoir.setZfactor(zFactor);

        ThresholdedRandomCutForest test = ThresholdedRandomCutForest.builder().dimensions(2).shingleSize(1)
                .randomSeed(77).timeDecay(timedecay).forestMode(ForestMode.DISTANCE).build();
        test.setZfactor(zFactor); // using the same apples to apples comparison

        String name = "clustering_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));

        Random noiseGen = new Random(randomSeed + 1);
        for (int degree = 0; degree < 360; degree += 1) {
            int index = 0;
            while (index < data.length) {
                boolean injected = false;
                float[] vec;
                if (noiseGen.nextDouble() < 0.005) {
                    injected = true;
                    double[] candAnomaly = new double[2];
                    // generate points along x axis
                    candAnomaly[0] = (range / 2 * noiseGen.nextDouble() + range / 2);
                    candAnomaly[1] = 0.1 * (2.0 * noiseGen.nextDouble() - 1.0);
                    int antiFan = noiseGen.nextInt(numberOfFans);
                    // rotate to be 90-180 degrees away -- these are decidedly anomalous
                    vec = toFloatArray(rotateClockWise(candAnomaly,
                            -2 * PI * (degree + 180 * (1 + 2 * antiFan) / numberOfFans) / 360));
                } else {
                    vec = toFloatArray(rotateClockWise(data[index], -2 * PI * degree / 360));
                    ++index;
                }

                GenericAnomalyDescriptor<float[]> result = reservoir.process(vec, 1.0f, null, true);

                AnomalyDescriptor res = test.process(toDoubleArray(vec), 0L);
                double grade = res.getAnomalyGrade();

                if (result.getRepresentativeList() != null) {
                    double sum = 0;
                    for (Weighted<float[]> rep : result.getRepresentativeList()) {
                        assert (rep.weight <= 1.0);
                        sum += rep.weight;
                    }
                    // checking likelihood summing to 1
                    assertEquals(sum, 1.0, 1e-6);
                }
                if (injected) {
                    if (result.getAnomalyGrade() > 0) {
                        ++truePos;
                    } else {
                        ++falseNeg;
                    }
                    if (grade > 0) {
                        ++truePosRCF;
                        assert (res.attribution != null);
                        // even though scoring is different, we should see attribution add up to score
                        assertEquals(res.attribution.getHighLowSum(), res.getRCFScore(), 1e-6);
                    } else {
                        ++falseNegRCF;
                    }
                } else {
                    if (result.getAnomalyGrade() > 0) {
                        ++falsePos;
                    }
                    if (grade > 0) {
                        ++falsePosRCF;
                        assert (res.attribution != null);
                        // even though scoring is different, we should see attribution add up to score
                        assertEquals(res.attribution.getHighLowSum(), res.getRCFScore(), 1e-6);
                    }
                }
            }

            if (falsePos + truePos == 0) {
                throw new IllegalStateException("");
            }

            checkArgument(falseNeg + truePos == falseNegRCF + truePosRCF, " incorrect accounting");
            System.out.println(" at degree " + degree + " injected " + (truePos + falseNeg));
            System.out.print("Precision = " + precision(truePos, falsePos));
            System.out.println(" Recall = " + recall(truePos, falseNeg));
            System.out.print("RCF Distance Mode Precision = " + precision(truePosRCF, falsePosRCF));
            System.out.println(" RCF Distance Mode Recall = " + recall(truePosRCF, falseNegRCF));

        }
        // attempting merge
        long number = new Random().nextLong();
        int size = reservoirSize - new Random().nextInt(100);
        double newShrinkage = new Random().nextDouble();
        int reps = new Random().nextInt(10);
        GlobalLocalAnomalyDetector.Builder builder = GlobalLocalAnomalyDetector.builder().capacity(size)
                .shrinkage(newShrinkage).numberOfRepresentatives(reps).timeDecay(timedecay).randomSeed(number);
        GlobalLocalAnomalyDetector<float[]> newDetector = new GlobalLocalAnomalyDetector<>(reservoir, reservoir,
                builder, true, Summarizer::L1distance);
        assertEquals(newDetector.getCapacity(), size);
        assertNotEquals(newDetector.getClusters(), null);
        assertEquals(newDetector.numberOfRepresentatives, reps);
        assertEquals(newDetector.shrinkage, newShrinkage);
        assert (newDetector.getClusters() != null);
        float[] weight = newDetector.sampler.getWeightArray();
        for (int i = 0; i < size - 1; i += 2) {
            assert (weight[i] >= weight[i + 1]);
        }
        GlobalLocalAnomalyDetector<float[]> another = new GlobalLocalAnomalyDetector<>(reservoir, reservoir, builder,
                false, Summarizer::L2distance);
        assertNull(another.getClusters());
        file.close();
    }

    double precision(int truePos, int falsePos) {
        return (truePos + falsePos > 0) ? 1.0 * truePos / (truePos + falsePos) : 1.0;
    }

    double recall(int truePos, int falseNeg) {
        return (truePos + falseNeg > 0) ? 1.0 * truePos / (truePos + falseNeg) : 1.0;
    }

}
