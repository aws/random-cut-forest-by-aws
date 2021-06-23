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
import static java.lang.Math.cos;
import static java.lang.Math.sin;

import java.util.Random;

/**
 * This class samples point from a mixture of 2 multi-variate normal
 * distribution with covariance matrices of the form sigma * I. One of the
 * normal distributions is considered the base distribution, the second is
 * considered the anomaly distribution, and there are random transitions between
 * the two.
 */
public class ExampleDataSets {

    public static double[][] generateFan(int numberPerBlade, int numberOfBlades) {
        if ((numberOfBlades > 12) || (numberPerBlade <= 0))
            return null;
        int newDimensions = 2;
        int dataSize = numberOfBlades * numberPerBlade;

        Random prg = new Random(0);
        NormalMixtureTestData generator = new NormalMixtureTestData(0.0, 1.0, 0.0, 1.0, 0, 1);
        double[][] data = generator.generateTestData(dataSize, newDimensions, 100);

        double[][] transformedData = new double[data.length][newDimensions];
        for (int j = 0; j < data.length; j++) {

            // shrink

            transformedData[j][0] = 0.05 * data[j][0];
            transformedData[j][1] = 0.2 * data[j][1];
            double toss = prg.nextDouble();

            // rotate
            int i = 0;
            while (i < numberOfBlades + 1) {
                if (toss < i * 1.0 / numberOfBlades) {
                    double[] vec = rotateClockWise(transformedData[j], 2 * PI * i / numberOfBlades);
                    transformedData[j][0] = vec[0] + 0.6 * sin(2 * PI * i / numberOfBlades);
                    transformedData[j][1] = vec[1] + 0.6 * cos(2 * PI * i / numberOfBlades);
                    break;
                } else
                    ++i;
            }
        }
        return transformedData;

    }

    public static double[] rotateClockWise(double[] point, double theta) {
        double[] result = new double[2];
        result[0] = cos(theta) * point[0] + sin(theta) * point[1];
        result[1] = -sin(theta) * point[0] + cos(theta) * point[1];
        return result;
    }

    public static double[][] generate(int size) {
        Random prg = new Random();
        double[][] data = new double[size][2];

        for (int i = 0; i < size; i++) {
            boolean test = false;
            while (!test) {
                double x = 2 * prg.nextDouble() - 1;
                double y = 2 * prg.nextDouble() - 1;
                if (x * x + y * y <= 1) {
                    if (y > 0) {
                        if (x > 0 && ((x - 0.5) * (x - 0.5) + y * y) <= 0.25) {
                            test = ((x - 0.5) * (x - 0.5) + y * y > 1.0 / 32) && (prg.nextDouble() < 0.6);
                        }
                    } else {
                        if (x > 0) {
                            if ((x - 0.5) * (x - 0.5) + y * y > 1.0 / 32) {
                                test = ((x - 0.5) * (x - 0.5) + y * y < 0.25) || (prg.nextDouble() < 0.4);
                            }
                        } else {
                            test = ((x + 0.5) * (x + 0.5) + y * y > 0.25) && (prg.nextDouble() < 0.2);
                        }
                    }
                }
                if (test) {
                    data[i][0] = x;
                    data[i][1] = y;
                }
            }
        }
        return data;
    }
}
