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
import java.util.Random;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;

public class LowNoisePeriodic implements Example {

    public static void main(String[] args) throws Exception {
        new LowNoisePeriodic().run();
    }

    @Override
    public String command() {
        return "Thresholded_Multi_Dim_example with low noise";
    }

    @Override
    public String description() {
        return "Thresholded Multi Dimensional Example with Low Noise";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 100000;
        int initialSegment = 100;

        double[] reference = new double[] { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 9.5f, 8.5f, 7.5f, 6.5f, 6.0f, 6.5f,
                7.0f, 7.5f, 9.5f, 11.0f, 12.5f, 10.5f, 8.5f, 7.0f, 5.0f, 3.0f, 2.0f, 1.0f };

        // the noise should leave suffient gap between the consecutive levels
        double noise = 0.25;
        // the noise will be amplified by something within [factorRange, 2*factorRange]
        // increase should lead to increased precision--recall; likewise decrease must
        // also
        // lead to decreased precision recall; if the factor = 1, then the anomalies are
        // information theoretically almost non-existent
        double anomalyFactor = 10;

        double slope = 0.2 * sampleSize
                * (Arrays.stream(reference).max().getAsDouble() - Arrays.stream(reference).min().getAsDouble()) / 50000;

        // to analyse without linear shift; comment out the line below and change the
        // slope above as desired
        slope = 0;

        double anomalyRate = 0.005;
        long seed = new Random().nextLong();
        System.out.println(" Seed " + seed);
        Random rng = new Random(seed);
        int numAnomalies = 0;
        int incorrectlyFlagged = 0;
        int correct = 0;
        int late = 0;

        // change the transformation below to experiment;
        // if slope != 0 then NONE will have poor result
        // both of the difference operations also introduce many errors
        TransformMethod method = TransformMethod.NORMALIZE;

        int dimensions = shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).anomalyRate(0.01).forestMode(ForestMode.STANDARD).startNormalization(32)
                .transformMethod(method).outputAfter(32).initialAcceptFraction(0.125)
                // for 1D data weights should not alter results significantly (if in reasonable
                // range say [0.1,10]
                // weights are not recommended for 1D, but retained here for illustration
                // as well as a mechanism to verify that results do not vary significantly
                .weights(new double[] { 1.0 })
                // change to transformDecay( 1.0/(desired interval length)) to perform
                // a moving average smoothing the default is 1.0/sampleSize
                // .transformDecay(1.0/sampleSize)
                .build();

        // the following ignore anomalies that are shifted up or down by a fixed amount
        // from the internal prediction of RCF. Default is 0.001

        // the below will show results like
        // missed current value 3.0 (say X), intended 1.0 (equiv., X - noise), because
        // the shift up in the actual was not 2*noise

        // forest.setIgnoreNearExpectedFromAbove( new double [] {2*noise});

        // or to suppress all anomalies that are shifted up from predicted
        // for any sequence; using Double.MAX_VALUE may cause overflow
        // forest.setIgnoreNearExpectedFromAbove(new double [] {Float.MAX_VALUE});

        // the below will show results like
        // missed current value 5.5 (say Y), intended 7.5 (equiv., Y + noise) because
        // the shift down in the actual was not 2*noise, in effect we suppress all
        // anomalies

        // forest.setIgnoreNearExpectedFromBelow(new double [] {noise*2});

        // the following suppresses all anomalies that shifted down compared to
        // predicted
        // for any sequence

        // forest.setIgnoreNearExpectedFromBelow(new double [] {Float.MAX_VALUE});

        double[] value = new double[] { 0.0 };

        int lastAnomaly = 0;

        for (int count = 0; count < dataSize; count++) {
            boolean anomaly = false;

            double intendedValue = reference[(count + 4) % reference.length] + slope * count;
            // extremely periodic signal -- note that there is no periodicity detection
            value[0] = intendedValue;
            if (rng.nextDouble() < anomalyRate && count > initialSegment) {
                double anomalyValue = noise * anomalyFactor * (1 + rng.nextDouble());
                value[0] += (rng.nextDouble() < 0.5) ? -anomalyValue : anomalyValue;
                anomaly = true;
                ++numAnomalies;
            } else {
                value[0] += (2 * rng.nextDouble() - 1) * noise;
            }

            AnomalyDescriptor result = forest.process(new double[] { value[0] }, 0);

            if (result.getAnomalyGrade() > 0) {
                System.out.print(count + " " + result.getAnomalyGrade() + " ");
                if (result.getRelativeIndex() < 0) {
                    System.out.print((lastAnomaly == count + result.getRelativeIndex()) + " "
                            + (-result.getRelativeIndex()) + " steps ago,");
                    if (lastAnomaly == count + result.getRelativeIndex()) {
                        late++;
                    } else {
                        incorrectlyFlagged++;
                    }
                } else {
                    System.out.print(anomaly);
                    if (anomaly) {
                        correct++;
                    } else {
                        incorrectlyFlagged++;
                    }
                }
                System.out.print(" current value " + value[0]);
                if (result.isExpectedValuesPresent()) {
                    System.out.print(" expected " + result.getExpectedValuesList()[0][0] + " instead of  "
                            + result.getPastValues()[0]);
                }
                System.out.print(" score " + result.getRCFScore() + " threshold " + result.getThreshold());
                System.out.println();
            } else if (anomaly) {
                System.out.println(count + " missed current value " + value[0] + ", intended " + intendedValue
                        + ", score " + result.getRCFScore() + ", threshold " + result.getThreshold());

            }
            if (anomaly) {
                lastAnomaly = count;
            }
        }

        System.out.println("Anomalies " + numAnomalies + ",  correct " + correct + ", late " + late
                + ", incorrectly flagged " + incorrectlyFlagged);

    }

}
