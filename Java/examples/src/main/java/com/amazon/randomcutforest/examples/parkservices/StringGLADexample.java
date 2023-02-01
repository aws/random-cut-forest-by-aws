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

package com.amazon.randomcutforest.examples.parkservices;

import static java.lang.Math.min;

import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.GlobalLocalAnomalyDetector;
import com.amazon.randomcutforest.parkservices.returntypes.GenericAnomalyDescriptor;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.util.Weighted;

/**
 * A clustering based anomaly detection for strings for two characters using
 * edit distance. Note that the algorithm does not have any inbuilt test for
 * verifying if the distance is indeed a metric (other than checking for
 * non-negative values.
 */
public class StringGLADexample implements Example {

    public static void main(String[] args) throws Exception {
        new StringGLADexample().run();
    }

    @Override
    public String command() {
        return "Clustering based Global-Local Anomaly Detection Example for strings";
    }

    @Override
    public String description() {
        return "Clustering based Global-Local Anomaly Detection Example for strings";
    }

    @Override
    public void run() throws Exception {
        long seed = new Random().nextLong();
        System.out.println("seed : " + seed);
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
        boolean printClusters = true;
        boolean printFalseNeg = false;
        boolean printFalsePos = false;
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
        // for non-geometric bounded distances, such as for strings, keep the factor at
        // 3.0 or below
        // minimum is 2.5, set as default; uncomment to change
        // reservoir.setZfactor(DEFAULT_Z_FACTOR);

        int truePos = 0;
        int falsePos = 0;
        int falseNeg = 0;
        for (int y = 0; y < points.length; y++) {

            GenericAnomalyDescriptor<char[]> result = reservoir.process(points[y], 1.0f, null, true);
            if (result.getAnomalyGrade() > 0) {
                if (!injected[y]) {
                    ++falsePos;
                    List<Weighted<char[]>> list = result.getRepresentativeList();
                    if (printFalsePos) {
                        System.out.println(result.getScore() + " " + injected[y] + " at " + y + " dist "
                                + dist.apply(points[y], list.get(0).index) + " " + result.getThreshold());
                        printCharArray(list.get(0).index);
                        System.out.println();
                        printCharArray(points[y]);
                        System.out.println();
                    }
                } else {
                    ++truePos;
                }
            } else if (injected[y]) {
                ++falseNeg;
                if (printFalseNeg) {
                    System.out.println(" missed " + result.getScore() + "  " + result.getThreshold());
                }
            }

            if (printClusters && y % 10000 == 0 && y > 0) {
                System.out.println(" at " + y);
                printClusters(reservoir.getClusters());

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

    public static void printCharArray(char[] a) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == '-') {
                System.out.print(ANSI_RED + a[i] + ANSI_RESET);
            } else {
                System.out.print(ANSI_BLUE + a[i] + ANSI_RESET);
            }
        }

    }

    public void printClusters(List<ICluster<char[]>> summary) {
        for (int i = 0; i < summary.size(); i++) {
            double weight = summary.get(i).getWeight();
            System.out.println("Cluster " + i + " representatives, weight "
                    + ((float) Math.round(1000 * weight) * 0.001) + " avg radius " + summary.get(i).averageRadius());
            List<Weighted<char[]>> representatives = summary.get(i).getRepresentatives();
            for (int j = 0; j < representatives.size(); j++) {
                double t = representatives.get(j).weight;
                t = Math.round(1000.0 * t / weight) * 0.001;
                System.out
                        .print("relative weight " + (float) t + " length " + representatives.get(j).index.length + " ");
                printCharArray(representatives.get(j).index);
                System.out.println();
            }
            System.out.println();
        }
    }

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

    double precision(int truePos, int falsePos) {
        return (truePos + falsePos > 0) ? 1.0 * truePos / (truePos + falsePos) : 1.0;
    }

    double recall(int truePos, int falseNeg) {
        return (truePos + falseNeg > 0) ? 1.0 * truePos / (truePos + falseNeg) : 1.0;
    }

}
