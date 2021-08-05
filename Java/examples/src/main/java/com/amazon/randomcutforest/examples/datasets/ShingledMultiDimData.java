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

package com.amazon.randomcutforest.examples.datasets;

import java.util.Random;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.PI;

/**
 * Serialize a Random Cut Forest using the
 * <a href="https://github.com/protostuff/protostuff">protostuff</a> library.
 */
public class ShingledMultiDimData {

    public static double[][] generateShingledData(int size, int period, int shingleSize, int baseDimension, long seed) {
        double[][] answer = new double[size][];
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[][] history = new double [shingleSize] [];
        int count = 0;
        double[][] data = getMultiDimData(size + shingleSize - 1, period, 100, 5, seed, baseDimension);
        for (int j = 0; j < size + shingleSize - 1; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shingleSize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                // System.out.println("Adding " + j);
                answer[count++] = getShinglePoint(history, entryIndex, shingleSize,baseDimension);
            }
        }
        return answer;
    }

    private static double[] getShinglePoint(double[][] recentPointsSeen, int indexOfOldestPoint, int shingleLength, int baseDimension) {
        double[] shingledPoint = new double[shingleLength*baseDimension];
        int count = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double[] point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            checkArgument(point.length == baseDimension, "error in point set");
            for(int i=0;i<baseDimension;i++) {
                shingledPoint[count++] = point[i];
            }
        }
        return shingledPoint;
    }

    private static double[][] getMultiDimData(int num, int period, double amplitude, double noise, long seed, int baseDimension) {

        double[][] data = new double[num][];
        Random prg = new Random(seed);
        Random noiseprg = new Random(prg.nextLong());
        double [] phase = new double[baseDimension];
        double [] amp = new double [baseDimension];
        for(int i=0;i<baseDimension;i++){
            phase[i] = prg.nextInt(period);
            amp[i] = (1 + 0.2 * prg.nextDouble())*amplitude;
        }

        for (int i = 0; i < num; i++) {
            data[i] = new double[baseDimension];
            for(int j = 0;j <baseDimension;j++) {
                data[i][j] = amp[j] * Math.cos(2 * PI * (i + phase[j]) / period) + noise * noiseprg.nextDouble();
                if (noiseprg.nextDouble() < 0.01) {
                    data[i][j] += noiseprg.nextDouble() < 0.5 ? 10 * noise : -10 * noise;
                }
            }
        }

        return data;
    }
}
