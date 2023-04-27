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

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;

import java.util.Arrays;
import java.util.Random;

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

        int shingleSize = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 100000;
        int initialSegment = 100;

        double[] reference = new double[] { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 9.5f, 8.5f, 7.5f, 6.5f, 6.0f, 6.5f,
                7.0f, 7.5f, 9.5f, 11.0f, 12.5f, 10.5f, 8.5f, 7.0f, 5.0f, 3.0f, 2.0f, 1.0f };

        // change this to control the percent deviation
        // NOTE that if the noise is smaller than 0.003 times the actual value then
        // it would be difficult to detect the anomalies unless the slope is 0

        double noise = 5.0;
        // maximum of reference
        double slope = 0.2 * sampleSize
                * (Arrays.stream(reference).max().getAsDouble() - Arrays.stream(reference).min().getAsDouble()) / 50000;

        // to analyse without linear shift
        // slope = 0;

        double anomalyRate = 0.005;
        long seed = new Random().nextLong();
        System.out.println(" Seed " + seed);
        Random rng = new Random(seed);
        int numAnomalies = 0;
        int incorrectlyFlagged = 0;
        int correct = 0;
        int late = 0;

        // change the transformation below
        TransformMethod method = TransformMethod.NORMALIZE;
        int dimensions = shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .internalShinglingEnabled(true).anomalyRate(0.01).forestMode(ForestMode.STANDARD).startNormalization(32)
                .transformMethod(method).outputAfter(32).initialAcceptFraction(0.125)
                // for 1D data weights should not alter results significantly (if in reasonable
                // range say [0.1,10]
                // weights are not recommended for 1D, but retained here for illustration
                // as well as a mechanism to verify that results do not vary significantly
                .weights(new double[] { 1.0 }).build();

        // the following ignore anomalies that are shifted up or down by a fixed amount
        // from the internal prediction of RCF. Default is 0.001

        // the below will show results like
        // missed current value 3.0 (say X), intended 1.0 (equiv., X - noise), because
        // the shift up in the actual was not 2*noise

        // forest.setIgnoreNearExpectedFromAbove(2 * noise);

        // or to suppress all anomalies that are shifted up from predicted
        // for any sequence
        // forest.setIgnoreNearExpectedFromAbove(Float.MAX_VALUE);

        // the below will show results like
        // missed current value 5.5 (say Y), intended 7.5 (equiv., Y + noise) because
        // the shift down in the actual was not 2*noise, in effect we suppress all
        // anomalies

        // forest.setIgnoreNearExpectedFromBelow(2*noise);

        // the following suppresses all anomalies that shifted down compared to
        // predicted
        // for any sequence

        // forest.setIgnoreNearExpectedFromBelow(Float.MAX_VALUE);

        double[] value = new double[] { 0.0 };

        int lastAnomaly = 0;

        for (int count = 0; count < dataSize; count++) {
            boolean anomaly = false;

            double intendedValue = reference[(count + 4) % reference.length] + slope * count;
            // extremely periodic signal
            value[0] = intendedValue;
            if (rng.nextDouble() < anomalyRate && count > initialSegment) {
                value[0] += (rng.nextDouble() < 0.5) ? -noise : noise;
                anomaly = true;
                ++numAnomalies;
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
