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

import java.util.Random;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedPredictive implements Example {

    public static void main(String[] args) throws Exception {
        new com.amazon.randomcutforest.examples.parkservices.ThresholdedPredictive().run();
    }

    @Override
    public String command() {
        return "Thresholded_Predictive_example";
    }

    @Override
    public String description() {
        return "Example of predictive forecast across multiple time series using ThresholdedRCF";
    }

    @Override
    public void run() throws Exception {

        int sampleSize = 256;
        int baseDimensions = 1;
        int length = 4 * sampleSize;
        int outputAfter = 128;

        long seed = 2022L;
        Random random = new Random(seed);
        int numberOfModels = 10;
        MultiDimDataWithKey[] dataWithKeys = new MultiDimDataWithKey[numberOfModels];
        ThresholdedRandomCutForest[] forests = new ThresholdedRandomCutForest[numberOfModels];
        int[] period = new int[numberOfModels];

        double alertThreshold = 300;
        double lastActualSum = 0;

        int anomalies = 0;
        for (int k = 0; k < numberOfModels; k++) {
            period[k] = (int) Math.round(40 + 30 * random.nextDouble());
            dataWithKeys[k] = ShingledMultiDimDataWithKeys.getMultiDimData(length, period[k], 100, 10, seed,
                    baseDimensions, false);
            anomalies += dataWithKeys[k].changes.length;
        }

        System.out.println(anomalies + " anomalies injected ");

        int shingleSize = 10;
        int horizon = 20;
        for (int k = 0; k < numberOfModels; k++) {
            forests[k] = new ThresholdedRandomCutForest.Builder<>().compact(true)
                    .dimensions(baseDimensions * shingleSize).precision(Precision.FLOAT_32).randomSeed(seed + k)
                    .internalShinglingEnabled(true).shingleSize(shingleSize).outputAfter(outputAfter)
                    .transformMethod(TransformMethod.NORMALIZE).build();

        }

        boolean predictNextCrossing = true;
        boolean actualCrossingAlerted = false;

        boolean printPredictions = false;
        boolean printEvents = true;

        for (int i = 0; i < length; i++) {
            double[] prediction = new double[horizon];

            // any prediction needs suffient data
            // it's best to suggest 0 till such
            if (i > sampleSize) {
                for (int k = 0; k < numberOfModels; k++) {
                    RangeVector forecast = forests[k].extrapolate(horizon).rangeVector;
                    for (int t = 0; t < horizon; t++) {
                        prediction[t] += forecast.values[t];
                    }
                }
                if (prediction[horizon - 1] > alertThreshold && predictNextCrossing) {
                    if (printEvents) {
                        System.out.println("Currently at " + i + ", should cross " + alertThreshold + " at sequence "
                                + (i + horizon - 1));
                    }
                    predictNextCrossing = false;
                } else if (prediction[horizon - 1] < alertThreshold && !predictNextCrossing) {
                    predictNextCrossing = true;
                }
                if (printPredictions) {
                    for (int t = 0; t < horizon; t++) {
                        System.out.println((i + t) + " " + prediction[t]);
                    }
                    System.out.println();
                    System.out.println();
                }
            }

            // now look at actuals
            double sumValue = 0;
            for (int k = 0; k < numberOfModels; k++) {
                sumValue += dataWithKeys[k].data[i][0];
            }
            if (lastActualSum > alertThreshold && sumValue > alertThreshold) {
                if (!actualCrossingAlerted) {
                    if (printEvents) {
                        System.out.println(" Crossing " + alertThreshold + " at consecutive sequence indices " + (i - 1)
                                + " " + i);
                    }
                    actualCrossingAlerted = true;
                }
            } else if (sumValue < alertThreshold) {
                actualCrossingAlerted = false;
            }
            lastActualSum = sumValue;

            // update model
            for (int k = 0; k < numberOfModels; k++) {
                forests[k].process(dataWithKeys[k].data[i], 0L);
            }
        }

    }

}