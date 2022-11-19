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

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;
import com.amazon.randomcutforest.parkservices.RCFCaster;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

/**
 * The following example demonstrates the self calibration of RCFCast. Change
 * various parameters -- we recommend keeping baseDimension = 1 (for single
 * variate forecasting -- multivariate forecasting can be a complicated
 * endeavour. The value shifForViz is for easier visualization.
 *
 * Once the datafile calibration_example is produced consider plotting it. For
 * example to use gnuplot, to generate a quick and dirty gif file, consider
 * these commands set terminal gif transparent animate delay 5 set output
 * "example.gif" do for [i = 0:3000:3] { (all the below in a single line) plot
 * [0:1000][-100:500] "example" i 0 u 1:2 w l lc "black" t "Data (seen one at a
 * time)", "example" index (i+3) u 1:2 w l lw 2 lc "blue" t " Online Forecast
 * (future)", "example" i (i+2) u 1:(100*$8) w l lw 2 lc "magenta" t "Interval
 * Accuracy %", "example" index (i+3) u 1:($4-$2):($3-$4) w filledcurves fc
 * "blue" fs transparent solid 0.3 noborder t "Calibrated uncertainty range
 * (future)", "example" index (i+2) u 1:7:6 w filledcurves fc "brown" fs
 * transparent solid 0.5 noborder t "Observed error distribution range (past)",
 * "example" i (i+1) u 1:2 w impulses t "", 0 lc "gray" t "", 100 lc "gray" t
 * "", 80 lc "gray" t"" }
 *
 * Try different calibrations below to see the precision over the intervals. The
 * struggle of past and new data would become obvious; however the algorithm
 * would self-calibrate eventually. Changing the different values for
 * transformDecay() would correspond to different moving average analysis.
 *
 */
public class RCFCasterExample implements Example {

    public static void main(String[] args) throws Exception {
        new RCFCasterExample().run();
    }

    @Override
    public String command() {
        return "Calibrated RCFCast";
    }

    @Override
    public String description() {
        return "Calibrated RCFCast Example";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 2 * sampleSize;

        // Multi attribute forecasting is less understood than singe attribute
        // forecasting;
        // it is not always clear or easy to decide if multi-attribute forecasting is
        // reasonable
        // but the code below will run for multi-attribute case.
        int baseDimensions = 2;
        int forecastHorizon = 15;
        int shingleSize = 20;
        int outputAfter = 64;

        long seed = 2023L;

        double[][] fulldata = new double[2 * dataSize][];
        double shiftForViz = 200;
        System.out.println("seed = " + seed);
        // change the last argument seed for a different run
        MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 50, 50, 5, seed,
                baseDimensions, true);
        for (int i = 0; i < dataSize; i++) {
            fulldata[i] = Arrays.copyOf(dataWithKeys.data[i], baseDimensions);
            fulldata[i][0] += shiftForViz;
        }

        // changing both period and amplitude for fun
        MultiDimDataWithKey second = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize, 70, 30, 5, seed + 1,
                baseDimensions, true);
        for (int i = 0; i < dataSize; i++) {
            fulldata[dataSize + i] = Arrays.copyOf(second.data[i], baseDimensions);
            fulldata[dataSize + i][0] += shiftForViz;
        }

        int dimensions = baseDimensions * shingleSize;
        // change this line to try other transforms; but the default is NORMALIZE
        // uncomment the transformMethod() below
        TransformMethod transformMethod = TransformMethod.NORMALIZE;
        RCFCaster caster = RCFCaster.builder().dimensions(dimensions).randomSeed(seed + 1).numberOfTrees(numberOfTrees)
                .shingleSize(shingleSize).sampleSize(sampleSize).internalShinglingEnabled(true).precision(precision)
                .anomalyRate(0.01).outputAfter(outputAfter).calibration(Calibration.MINIMAL)
                // the following affects the moving average in many of the transformations
                // the 0.02 corresponds to a half life of 1/0.02 = 50 observations
                // this is different from the timeDecay() of RCF; however it is a similar
                // concept
                .transformDecay(0.02).forecastHorizon(forecastHorizon).initialAcceptFraction(0.125).build();

        String name = "example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));

        for (int j = 0; j < fulldata.length; j++) {
            file.append(j + " ");
            for (int k = 0; k < baseDimensions; k++) {
                file.append(fulldata[j][k] + " ");
            }
            file.append("\n");
        }
        file.append("\n");
        file.append("\n");

        for (int j = 0; j < fulldata.length; j++) {
            ForecastDescriptor result = caster.process(fulldata[j], 0L);
            printResult(file, result, j, baseDimensions);
        }
        file.close();

    }

    void printResult(BufferedWriter file, ForecastDescriptor result, int current, int inputLength) throws IOException {
        RangeVector forecast = result.getTimedForecast().rangeVector;
        float[] errorP50 = result.getObservedErrorDistribution().values;
        float[] upperError = result.getObservedErrorDistribution().upper;
        float[] lowerError = result.getObservedErrorDistribution().lower;
        DiVector rmse = result.getErrorRMSE();
        float[] mean = result.getErrorMean();
        float[] calibration = result.getCalibration();

        file.append(current + " " + 1000 + "\n");
        file.append("\n");
        file.append("\n");

        // block corresponding to the past; print the errors
        for (int i = forecast.values.length / inputLength - 1; i >= 0; i--) {
            file.append((current - i) + " ");
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                file.append(mean[k] + " " + rmse.high[k] + " " + rmse.low[k] + " " + errorP50[k] + " " + upperError[k]
                        + " " + lowerError[k] + " " + calibration[k] + " ");
            }
            file.append("\n");
        }
        file.append("\n");
        file.append("\n");

        // block corresponding to the future; the projections and the projected errors
        for (int i = 0; i < forecast.values.length / inputLength; i++) {
            file.append((current + i) + " ");
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                file.append(forecast.values[k] + " " + forecast.upper[k] + " " + forecast.lower[k] + " ");
            }
            file.append("\n");
        }
        file.append("\n");
        file.append("\n");
    }

}
