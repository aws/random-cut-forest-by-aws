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

import static java.lang.Math.PI;

import java.util.Arrays;
import java.util.Random;

public class ShingledMultiDimDataWithKeys {

    public static MultiDimDataWithKey generateShingledDataWithKey(int size, int period, int shingleSize,
            int baseDimension, long seed) {
        double[][] answer = new double[size][];
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[][] history = new double[shingleSize][];
        int count = 0;
        MultiDimDataWithKey dataWithKeys = getMultiDimData(size + shingleSize - 1, period, 100, 5, seed, baseDimension);
        for (int j = 0; j < size + shingleSize - 1; ++j) { // we stream here ....
            history[entryIndex] = dataWithKeys.data[j];
            entryIndex = (entryIndex + 1) % shingleSize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                answer[count++] = getShinglePoint(history, entryIndex, shingleSize, baseDimension);
            }
        }
        return new MultiDimDataWithKey(answer, dataWithKeys.changeIndices, dataWithKeys.changes);
    }

    private static double[] getShinglePoint(double[][] recentPointsSeen, int indexOfOldestPoint, int shingleLength,
            int baseDimension) {
        double[] shingledPoint = new double[shingleLength * baseDimension];
        int count = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double[] point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            for (int i = 0; i < baseDimension; i++) {
                shingledPoint[count++] = point[i];
            }
        }
        return shingledPoint;
    }

    public static MultiDimDataWithKey getMultiDimData(int num, int period, double amplitude, double noise, long seed,
            int baseDimension) {
        return getMultiDimData(num, period, amplitude, noise, seed, baseDimension, false);
    }

    public static MultiDimDataWithKey getMultiDimData(int num, int period, double amplitude, double noise, long seed,
            int baseDimension, boolean useSlope) {
        double[][] data = new double[num][];
        double[][] changes = new double[num][];
        int[] changedIndices = new int[num];
        int counter = 0;
        Random prg = new Random(seed);
        Random noiseprg = new Random(prg.nextLong());
        double[] phase = new double[baseDimension];
        double[] amp = new double[baseDimension];
        double[] slope = new double[baseDimension];

        for (int i = 0; i < baseDimension; i++) {
            phase[i] = prg.nextInt(period);
            amp[i] = (1 + 0.2 * prg.nextDouble()) * amplitude;
            if (useSlope) {
                slope[i] = (0.25 - prg.nextDouble() * 0.5) * amplitude / period;
            }
        }

        for (int i = 0; i < num; i++) {
            data[i] = new double[baseDimension];
            boolean flag = (noiseprg.nextDouble() < 0.01);
            double[] newChange = new double[baseDimension];
            boolean used = false;
            for (int j = 0; j < baseDimension; j++) {
                data[i][j] = amp[j] * Math.cos(2 * PI * (i + phase[j]) / period) + slope[j] * i
                        + noise * noiseprg.nextDouble();
                if (flag && noiseprg.nextDouble() < 0.3) {
                    double factor = 5 * (1 + noiseprg.nextDouble());
                    double change = noiseprg.nextDouble() < 0.5 ? factor * noise : -factor * noise;
                    data[i][j] += newChange[j] = change;
                    used = true;
                }
            }
            if (used) {
                changedIndices[counter] = i;
                changes[counter++] = newChange;
            }
        }

        return new MultiDimDataWithKey(data, Arrays.copyOf(changedIndices, counter), Arrays.copyOf(changes, counter));
    }
}
