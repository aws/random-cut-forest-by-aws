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

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static java.lang.Math.abs;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

/**
 * centroidal clustering fails in many scenarios; primarily because a single
 * point in combination with a distance metric can only represent a sphere. A
 * reasonable solution is to use multiple well scattered centroids to represent
 * a cluster and has been long in use, see CURE
 * https://en.wikipedia.org/wiki/CURE_algorithm
 *
 * The following example demonstrates the use of a multicentroid clustering; the
 * data corresponds to 2*d clusters in d dimensions (d chosen randomly) such
 * that the clusters almost touch, but remain separable. Note that the knowledge
 * of the true number of clusters is not required -- the clustering is invoked
 * with a maximum of 5*d potential clusters, and yet the example often finds the
 * true 2*d clusters.
 */
public class RCFMultiSummarizeExample implements Example {

    public static void main(String[] args) throws Exception {
        new com.amazon.randomcutforest.examples.summarization.RCFMultiSummarizeExample().run();
    }

    @Override
    public String command() {
        return "RCF_Multi_Summarize_Example";
    }

    @Override
    public String description() {
        return "Example of using RCF Multi Summarization";
    }

    @Override
    public void run() throws Exception {
        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        int dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);

        double epsilon = 0.01;
        List<ICluster<float[]>> summary = Summarizer.multiSummarize(points, 5 * newDimensions, 0.1, true, 5,
                random.nextLong());
        System.out.println(summary.size() + " clusters for " + newDimensions + " dimensions, seed : " + seed);
        double weight = summary.stream().map(e -> e.getWeight()).reduce(Double::sum).get();
        System.out.println(
                "Total weight " + ((float) Math.round(weight * 1000) * 0.001) + " rounding to multiples of " + epsilon);
        System.out.println();

        for (int i = 0; i < summary.size(); i++) {
            double clusterWeight = summary.get(i).getWeight();
            System.out.println(
                    "Cluster " + i + " representatives, weight " + ((float) Math.round(1000 * clusterWeight) * 0.001));
            List<Weighted<float[]>> representatives = summary.get(i).getRepresentatives();
            for (int j = 0; j < representatives.size(); j++) {
                double t = representatives.get(j).weight;
                t = Math.round(1000.0 * t / clusterWeight) * 0.001;
                System.out.print("relative weight " + (float) t + " center (approx)  ");
                printArray(representatives.get(j).index, epsilon);
                System.out.println();
            }
            System.out.println();
        }

    }

    void printArray(float[] values, double epsilon) {
        System.out.print(" [");
        if (abs(values[0]) < epsilon) {
            System.out.print("0");
        } else {
            if (epsilon <= 0) {
                System.out.print(values[0]);
            } else {
                long t = (int) Math.round(values[0] / epsilon);
                System.out.print(t * epsilon);
            }
        }
        for (int i = 1; i < values.length; i++) {
            if (abs(values[i]) < epsilon) {
                System.out.print(", 0");
            } else {
                if (epsilon <= 0) {
                    System.out.print(", " + values[i]);
                } else {
                    long t = Math.round(values[i] / epsilon);
                    System.out.print(", " + t * epsilon);
                }
            }
        }
        System.out.print("]");
    }

    public float[][] getData(int dataSize, int newDimensions, int seed, BiFunction<float[], float[], Double> distance) {
        double baseMu = 0.0;
        double baseSigma = 1.0;
        double anomalyMu = 0.0;
        double anomalySigma = 1.0;
        double transitionToAnomalyProbability = 0.0;
        // ignoring anomaly cluster for now
        double transitionToBaseProbability = 1.0;
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

}