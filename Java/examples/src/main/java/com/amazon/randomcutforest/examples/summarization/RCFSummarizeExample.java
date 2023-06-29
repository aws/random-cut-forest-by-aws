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

import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static java.lang.Math.abs;

/**
 * The following example is based off a test of summarization and provides an
 * example use of summarization based on centroidal representation. The
 * clustering takes a distance function from (float[],float []) into double as
 * input, along with a maximum number of allowed clusters and provides a summary
 * which contains the list of cluster centers as "typical points" along with
 * relative likelihood.
 *
 * The specific example below corresponds to 2*d clusters (one each in +ve and
 * -ve axis for each of the d dimensions) where d is chosen at random between 3
 * and 13. The clusters are designed to almost touch -- but are separable (with
 * high probability) and should be discoverable separately. Note that the
 * algorithm does not require the knowledge of the true number of clusters (2*d)
 * but is run with a maximum allowed number 5*d.
 */
public class RCFSummarizeExample implements Example {

    public static void main(String[] args) throws Exception {
        new com.amazon.randomcutforest.examples.summarization.RCFSummarizeExample().run();
    }

    @Override
    public String command() {
        return "RCF_Summarize_Example";
    }

    @Override
    public String description() {
        return "Example of using RCF Summarization";
    }

    @Override
    public void run() throws Exception {
        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        int dataSize = 200000;

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);

        SampleSummary summary = Summarizer.l2summarize(points, 5 * newDimensions,42);
        System.out.println(
                summary.summaryPoints.length + " clusters for " + newDimensions + " dimensions, seed : " + seed);
        double epsilon = 0.01;
        System.out.println("Total weight " + summary.weightOfSamples + " rounding to multiples of " + epsilon);
        System.out.println();
        for (int i = 0; i < summary.summaryPoints.length; i++) {
            long t = Math.round(summary.relativeWeight[i] / epsilon);
            System.out.print("Cluster " + i + " relative weight " + ((float) t * epsilon) + " center (approx): ");
            printArray(summary.summaryPoints[i], epsilon);
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
                System.out.print((float) t * epsilon);
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
                    System.out.print(", " + ((float) t * epsilon));
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