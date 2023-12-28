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

import static java.lang.Math.min;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

public class ThresholdedForecast implements Example {

    public static void main(String[] args) throws Exception {
        new com.amazon.randomcutforest.examples.parkservices.ThresholdedForecast().run();
    }

    @Override
    public String command() {
        return "Thresholded_Forecast_example";
    }

    @Override
    public String description() {
        return "Example of Forecast using Thresholded RCF";
    }

    @Override
    public void run() throws Exception {

        int sampleSize = 256;
        int baseDimensions = 1;

        long seed = 100L;

        int length = 4 * sampleSize;
        int outputAfter = 128;

        // as the ratio of amplitude (signal) to noise is changed, the estimation range
        // in forecast
        // (or any other inference) should increase
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 10, seed,
                baseDimensions, true);
        System.out.println(dataWithKeys.changes.length + " anomalies injected ");

        // horizon/lookahead can be larger than shingleSize for transformations that do
        // not
        // involve differencing -- but longer horizon would have larger error
        int horizon = 60;
        int shingleSize = 30;

        // if the useSlope is set as true then it is recommended to use NORMALIZE or
        // SUBTRACT_MA as
        // transformation methods to adjust to the linear drift

        ThresholdedRandomCutForest forest = new ThresholdedRandomCutForest.Builder<>().compact(true)
                .dimensions(baseDimensions * shingleSize).precision(Precision.FLOAT_32).randomSeed(seed)
                .internalShinglingEnabled(true).shingleSize(shingleSize).outputAfter(outputAfter)
                .transformMethod(TransformMethod.NORMALIZE).build();

        if (forest.getTransformMethod() == TransformMethod.NORMALIZE_DIFFERENCE
                || forest.getTransformMethod() == TransformMethod.DIFFERENCE) {
            // single step differencing will not produce stable forecasts over long horizons
            horizon = min(horizon, shingleSize / 2 + 1);
        }
        double[] error = new double[horizon];
        double[] lowerError = new double[horizon];
        double[] upperError = new double[horizon];

        for (int j = 0; j < dataWithKeys.data.length; j++) {
            // forecast first; change centrality to achieve a control over the sampling
            // setting centrality = 0 would correspond to random sampling from the leaves
            // reached by
            // impute visitor

            // the following prints
            // <sequenceNo> <predicted_next_value> <likely_upper_bound> <likely_lower_bound>
            // where the sequence number varies between next-to-be-read .. (next + horizon
            // -1 )
            //
            // Every new element corresponds to a new set of horizon forecasts; we measure
            // the
            // errors keeping the leadtime fixed.
            //
            // verify that forecast is done before seeing the actual value (in the process()
            // function)
            //

            TimedRangeVector extrapolate = forest.extrapolate(horizon, true, 1.0);
            RangeVector forecast = extrapolate.rangeVector;
            for (int i = 0; i < horizon; i++) {
                System.out.println(
                        (j + i) + " " + forecast.values[i] + " " + forecast.upper[i] + " " + forecast.lower[i]);
                // compute errors
                if (j > outputAfter + shingleSize - 1 && j + i < dataWithKeys.data.length) {
                    double t = dataWithKeys.data[j + i][0] - forecast.values[i];
                    error[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.lower[i];
                    lowerError[i] += t * t;
                    t = dataWithKeys.data[j + i][0] - forecast.upper[i];
                    upperError[i] += t * t;
                }
            }
            System.out.println();
            System.out.println();
            forest.process(dataWithKeys.data[j], j);
        }

        System.out.println(forest.getTransformMethod().name() + " RMSE (as horizon increases) ");
        for (int i = 0; i < horizon; i++) {
            double t = error[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Lower (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = lowerError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();
        System.out.println("RMSE Upper (as horizon increases)");
        for (int i = 0; i < horizon; i++) {
            double t = upperError[i] / (dataWithKeys.data.length - shingleSize + 1 - outputAfter - i);
            System.out.print(Math.sqrt(t) + " ");
        }
        System.out.println();

    }

}