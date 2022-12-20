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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_NUMBER_OF_TREES;
import static com.amazon.randomcutforest.RandomCutForest.DEFAULT_SAMPLE_SIZE;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.calibration.Calibration;
import com.amazon.randomcutforest.parkservices.returntypes.AnalysisDescriptor;

public class SequentialAnalysis {

    /**
     * provides a list of anomnalies given a block of data. While this is a fairly
     * simple function, it is provided as a reference such that users do not have
     * depend on interpretations of sequentian analysis
     * 
     * @param data            the array containing the values
     * @param shingleSize     shinglesize of RCF
     * @param sampleSize      sampleSize of RCF
     * @param numberOfTrees   the numberOfTres used by RCF
     * @param timeDecay       the time decay parameter of RCF; think of half life of
     *                        data
     * @param outputAfter     the value after which we
     * @param transformMethod the transformation used in preprocessing
     * @param transformDecay  the half life of data in preprocessing (if in doubt,
     *                        use the same as timeDecay)
     * @param seed            a random seed
     * @return a list of anomalies
     */
    public static List<AnomalyDescriptor> detectAnomalies(double[][] data, int shingleSize, int sampleSize,
            int numberOfTrees, double timeDecay, int outputAfter, TransformMethod transformMethod,
            double transformDecay, long seed) {
        checkArgument(data != null, "cannot be a null array");
        int inputDimension = data[0].length;
        int dimensions = inputDimension * shingleSize;
        double fraction = 1.0 * outputAfter / sampleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(seed)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).anomalyRate(0.01).forestMode(ForestMode.STANDARD).timeDecay(timeDecay)
                .transformMethod(transformMethod).outputAfter(outputAfter).transformDecay(transformDecay)
                .initialAcceptFraction(fraction).build();
        ArrayList<AnomalyDescriptor> answer = new ArrayList<>();
        for (double[] point : data) {
            AnomalyDescriptor result = forest.process(point, 0L);
            if (result.getAnomalyGrade() > 0) {
                answer.add(result);
            }
        }
        return answer;
    }

    public static List<AnomalyDescriptor> detectAnomalies(double[][] data, int shingleSize, int sampleSize,
            double timeDecay, TransformMethod transformMethod) {
        return detectAnomalies(data, shingleSize, sampleSize, DEFAULT_NUMBER_OF_TREES, timeDecay, sampleSize / 4,
                transformMethod, timeDecay, new Random().nextLong());
    }

    public static List<AnomalyDescriptor> detectAnomalies(double[][] data, int shingleSize, double timeDecay,
            TransformMethod transformMethod, double transformDecay) {
        return detectAnomalies(data, shingleSize, DEFAULT_SAMPLE_SIZE, DEFAULT_NUMBER_OF_TREES, timeDecay,
                DEFAULT_SAMPLE_SIZE / 4, transformMethod, transformDecay, new Random().nextLong());
    }

    /**
     * Same as the anomaly detector but provides a list of anomalies as well as a
     * calibrated (with testing) interval and forecasts.
     * 
     * @param inputArray      the input
     * @param shingleSize     shingle size of RCF
     * @param sampleSize      samplesize of RCF
     * @param timeDecay       timedecay of RCF
     * @param outputAfter     the input after which we perform score evaluation
     * @param transformMethod transformation method of preprocessing
     * @param transformDecay  the time decay of preprocessing
     * @param forecastHorizon the number of steps to forecast (during and at the
     *                        end)
     * @param errorHorizon    the number of steps to perform calibration (during the
     *                        sequence)
     * @param percentile      the percentile of error one is interested in
     *                        calibrating (we recommend 0.1)
     * @param seed            random seed
     * @return a list of anomalies and the final forecast wilh callibration
     */
    public static AnalysisDescriptor forecastWithAnomalies(double[][] inputArray, int shingleSize, int sampleSize,
            double timeDecay, int outputAfter, TransformMethod transformMethod, double transformDecay,
            int forecastHorizon, int errorHorizon, double percentile, Calibration calibration, long seed) {
        checkArgument(inputArray != null, " input cannot be null");
        int inputDimension = inputArray[0].length;
        int dimensions = shingleSize * inputDimension;
        int numberOfTrees = 50;
        double fraction = 1.0 * outputAfter / sampleSize;
        RCFCaster caster = RCFCaster.builder().dimensions(dimensions).randomSeed(seed).numberOfTrees(numberOfTrees)
                .shingleSize(shingleSize).sampleSize(sampleSize).internalShinglingEnabled(true).anomalyRate(0.01)
                .forestMode(ForestMode.STANDARD).timeDecay(timeDecay).transformMethod(transformMethod)
                .outputAfter(outputAfter).calibration(calibration).initialAcceptFraction(fraction)
                .forecastHorizon(forecastHorizon).transformDecay(transformDecay).errorHorizon(errorHorizon)
                .percentile(percentile).build();

        ArrayList<AnomalyDescriptor> descriptors = new ArrayList<>();
        ForecastDescriptor last = null;
        for (double[] input : inputArray) {
            ForecastDescriptor descriptor = caster.process(input, 0L);
            if (descriptor.getAnomalyGrade() > 0) {
                descriptors.add(descriptor);
            }
            last = descriptor;
        }
        return new AnalysisDescriptor(descriptors, last);
    }

    public static AnalysisDescriptor forecastWithAnomalies(double[][] inputArray, int shingleSize, int sampleSize,
            double timeDecay, TransformMethod transformMethod, int forecastHorizon, int errorHorizon, long seed) {
        return forecastWithAnomalies(inputArray, shingleSize, sampleSize, timeDecay, sampleSize / 4, transformMethod,
                timeDecay, forecastHorizon, errorHorizon, 0.1, Calibration.SIMPLE, seed);
    }

}
