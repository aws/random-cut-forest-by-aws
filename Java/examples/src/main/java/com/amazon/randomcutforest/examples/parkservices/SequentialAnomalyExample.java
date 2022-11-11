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

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.SequentialAnalysis;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class SequentialAnomalyExample implements Example {

    public static void main(String[] args) throws Exception {
        new SequentialAnomalyExample().run();
    }

    @Override
    public String command() {
        return "Sequential_analysis_example";
    }

    @Override
    public String description() {
        return "Sequential Analysis Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 4 * sampleSize;

        // change this to try different number of attributes,
        // this parameter is not expected to be larger than 5 for this example
        int baseDimensions = 2;

        long seed = new Random().nextLong();

        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50,
                100, 5, seed, baseDimensions);
        double timeDecay = 1.0 / (10 * sampleSize);

        List<AnomalyDescriptor> anomalies = SequentialAnalysis.detectAnomalies(dataWithKeys.data, shingleSize,
                sampleSize, timeDecay, TransformMethod.NONE);
        int keyCounter = 0;

        for (AnomalyDescriptor result : anomalies) {

            // first print the changes
            while (keyCounter < dataWithKeys.changeIndices.length
                    && dataWithKeys.changeIndices[keyCounter] <= result.getInternalTimeStamp()) {
                System.out.println("timestamp " + dataWithKeys.changeIndices[keyCounter] + " CHANGE "
                        + Arrays.toString(dataWithKeys.changes[keyCounter]));
                ++keyCounter;
            }

            if (result.getAnomalyGrade() != 0) {
                System.out.print("timestamp " + result.getInternalTimeStamp() + " RESULT value ");
                for (int i = 0; i < baseDimensions; i++) {
                    System.out.print(result.getCurrentInput()[i] + ", ");
                }
                System.out.print("score " + result.getRCFScore() + ", grade " + result.getAnomalyGrade() + ", ");
                if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                    System.out.print(-result.getRelativeIndex() + " step(s) ago, ");
                }
                if (result.isExpectedValuesPresent()) {
                    if (result.getRelativeIndex() != 0 && result.isStartOfAnomaly()) {
                        System.out.print("instead of ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getPastValues()[i] + ", ");
                        }
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                            if (result.getPastValues()[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print("( "
                                        + (result.getPastValues()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    } else {
                        System.out.print("expected ");
                        for (int i = 0; i < baseDimensions; i++) {
                            System.out.print(result.getExpectedValuesList()[0][i] + ", ");
                            if (result.getCurrentInput()[i] != result.getExpectedValuesList()[0][i]) {
                                System.out.print("( "
                                        + (result.getCurrentInput()[i] - result.getExpectedValuesList()[0][i]) + " ) ");
                            }
                        }
                    }
                } else {
                    System.out.print("insufficient data to provide expected values");
                }
                System.out.println();
            }

        }

    }

}
