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

package com.amazon.randomcutforest.examples.summarization;

import static java.lang.Math.min;

import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.util.Weighted;

/**
 * the following example showcases the use of RCF multi-summarization on generic
 * types R, when provided with a distance function from (R,R) into double. In
 * this example R correpsonds to Strings and the distance is EditDistance The
 * srings are genrated from two clusters one where character A (or '-' for viz)
 * occurs with probability 2/3 and anothewr where it occurs with probability 1/3
 * (and the character B or '_' occurs with probability 2/3)
 *
 * Clearly, and the following example makes it visual, multicentroid approach is
 * necessary.
 *
 * All the strings do not have the same length. Note that the summarization is
 * asked with a maximum of 10 clusters but the algorithm self-adjusts to 2
 * clusters.
 */
public class RCFStringSummarizeExample implements Example {

    public static void main(String[] args) throws Exception {
        new com.amazon.randomcutforest.examples.summarization.RCFStringSummarizeExample().run();
    }

    @Override
    public String command() {
        return "RCF_String_Summarize_Example";
    }

    @Override
    public String description() {
        return "Example of using RCF String Summarization, uses multi-centroid approach";
    }

    @Override
    public void run() throws Exception {

        long seed = -8436172895711381300L;
        new Random().nextLong();
        System.out.println("String summarization seed : " + seed);
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
                RCFStringSummarizeExample::toyDistance, nextSeed, true, 0.1, 5);
        System.out.println();
        for (int i = 0; i < summary.size(); i++) {
            double weight = summary.get(i).getWeight();
            System.out.println(
                    "Cluster " + i + " representatives, weight " + ((float) Math.round(1000 * weight) * 0.001));
            List<Weighted<String>> representatives = summary.get(i).getRepresentatives();
            for (int j = 0; j < representatives.size(); j++) {
                double t = representatives.get(j).weight;
                t = Math.round(1000.0 * t / weight) * 0.001;
                System.out.print(
                        "relative weight " + (float) t + " length " + representatives.get(j).index.length() + " ");
                printString(representatives.get(j).index);
                System.out.println();
            }
            System.out.println();
        }

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

    // colors
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_BLUE = "\u001B[34m";

    public static void printString(String a) {
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) == '-') {
                System.out.print(ANSI_RED + a.charAt(i) + ANSI_RESET);
            } else {
                System.out.print(ANSI_BLUE + a.charAt(i) + ANSI_RESET);
            }
        }

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

}