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

package com.amazon.randomcutforest.testutils;

import java.util.Arrays;
import java.util.Random;

/**
 * This class samples point from a mixture of 2 multi-variate normal
 * distribution with covariance matrices of the form sigma * I. One of the
 * normal distributions is considered the base distribution, the second is
 * considered the anomaly distribution, and there are random transitions between
 * the two.
 */
public class NormalMixtureTestData {

    private final double baseMu;
    private final double baseSigma;
    private final double anomalyMu;
    private final double anomalySigma;
    private final double transitionToAnomalyProbability;
    private final double transitionToBaseProbability;

    public NormalMixtureTestData(double baseMu, double baseSigma, double anomalyMu, double anomalySigma,
            double transitionToAnomalyProbability, double transitionToBaseProbability) {
        this.baseMu = baseMu;
        this.baseSigma = baseSigma;
        this.anomalyMu = anomalyMu;
        this.anomalySigma = anomalySigma;
        this.transitionToAnomalyProbability = transitionToAnomalyProbability;
        this.transitionToBaseProbability = transitionToBaseProbability;
    }

    public NormalMixtureTestData() {
        this(0.0, 1.0, 4.0, 2.0, 0.01, 0.3);
    }

    public NormalMixtureTestData(double baseMu, double anomalyMu) {
        this(baseMu, 1.0, anomalyMu, 2.0, 0.01, 0.3);
    }

    public double[][] generateTestData(int numberOfRows, int numberOfColumns) {
        return generateTestData(numberOfRows, numberOfColumns, 0);
    }

    public double[][] generateTestData(int numberOfRows, int numberOfColumns, int seed) {
        double[][] result = new double[numberOfRows][numberOfColumns];
        boolean anomaly = false;

        NormalDistribution dist;
        if (seed != 0)
            dist = new NormalDistribution(new Random(seed));
        else
            dist = new NormalDistribution(new Random());

        for (int i = 0; i < numberOfRows; i++) {
            if (!anomaly) {
                fillRow(result[i], dist, baseMu, baseSigma);
                if (Math.random() < transitionToAnomalyProbability) {
                    anomaly = true;
                }
            } else {
                fillRow(result[i], dist, anomalyMu, anomalySigma);
                if (Math.random() < transitionToBaseProbability) {
                    anomaly = false;
                }
            }
        }

        return result;
    }

    public MultiDimDataWithKey generateTestDataWithKey(int numberOfRows, int numberOfColumns, int seed) {
        double[][] resultData = new double[numberOfRows][numberOfColumns];
        int[] change = new int[numberOfRows];
        int numberOfChanges = 0;
        boolean anomaly = false;

        NormalDistribution dist;
        if (seed != 0)
            dist = new NormalDistribution(new Random(seed));
        else
            dist = new NormalDistribution(new Random());

        for (int i = 0; i < numberOfRows; i++) {
            if (!anomaly) {
                fillRow(resultData[i], dist, baseMu, baseSigma);
                if (Math.random() < transitionToAnomalyProbability) {
                    change[numberOfChanges++] = i + 1; // next item is different
                    anomaly = true;
                }
            } else {
                fillRow(resultData[i], dist, anomalyMu, anomalySigma);
                if (Math.random() < transitionToBaseProbability) {
                    anomaly = false;
                    change[numberOfChanges++] = i + 1; // next item is different
                }
            }
        }

        return new MultiDimDataWithKey(resultData, Arrays.copyOf(change, numberOfChanges), null);
    }

    private void fillRow(double[] row, NormalDistribution dist, double mu, double sigma) {
        for (int j = 0; j < row.length; j++) {
            row[j] = dist.nextDouble(mu, sigma);
        }
    }

    static class NormalDistribution {
        private final Random rng;
        private final double[] buffer;
        private int index;

        NormalDistribution(Random rng) {
            this.rng = rng;
            buffer = new double[2];
            index = 0;
        }

        double nextDouble() {
            if (index == 0) {
                // apply the Box-Muller transform to produce Normal variates
                double u = rng.nextDouble();
                double v = rng.nextDouble();
                double r = Math.sqrt(-2 * Math.log(u));
                buffer[0] = r * Math.cos(2 * Math.PI * v);
                buffer[1] = r * Math.sin(2 * Math.PI * v);
            }

            double result = buffer[index];
            index = (index + 1) % 2;

            return result;
        }

        double nextDouble(double mu, double sigma) {
            return mu + sigma * nextDouble();
        }
    }
}
