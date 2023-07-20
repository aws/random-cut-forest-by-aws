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

package com.amazon.randomcutforest.returntypes;

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.function.BiFunction;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

public class SampleSummaryTest {

    /**
     * this class tests the return type data structure whereas
     * randomcutforest.SampleSummaryTest tests tha summarization algorithms.
     */
    int dataSize = 20000;
    int newDimensions = 2;
    Random random = new Random();

    @Test
    public void testConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(Collections.emptyList(), 0.6));

        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);
        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            // testing 0 weight
            weighted.add(new Weighted<>(point, 0.0f));
        }
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted, 1.3));

        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(0).weight = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(0).weight = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(0).weight = -1.0f;
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(0).weight = 1.0f;
        assertDoesNotThrow(() -> new SampleSummary(weighted));
        weighted.get(1).index = new float[newDimensions + 1];
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));

        weighted.get(1).index = new float[newDimensions];
        weighted.get(1).index[0] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(1).index[0] = Float.NEGATIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> new SampleSummary(weighted));
        weighted.get(1).index[0] = -1.0f;
        SampleSummary summary = new SampleSummary(weighted);
    }

    @Test
    public void addTypicalTest() {
        float[][] points = getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);
        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            // testing 0 weight
            weighted.add(new Weighted<>(point, 1.0f));
        }
        SampleSummary summary = new SampleSummary(weighted);
        assertThrows(IllegalArgumentException.class, () -> summary.addTypical(new float[1][2], new float[2]));
        assertDoesNotThrow(() -> summary.addTypical(new float[0][2], new float[0]));
        assertDoesNotThrow(() -> summary.addTypical(new float[2][4], new float[2]));
        assertThrows(IllegalArgumentException.class,
                () -> summary.addTypical(new float[][] { new float[2], new float[3] }, new float[2]));
        assertThrows(IllegalArgumentException.class,
                () -> summary.addTypical(new float[][] { new float[2], new float[3] }, new float[2]));
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
