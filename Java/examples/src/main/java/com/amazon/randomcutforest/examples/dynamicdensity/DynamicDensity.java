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

package com.amazon.randomcutforest.examples.dynamicdensity;

import static com.amazon.randomcutforest.testutils.ExampleDataSets.generate;
import static com.amazon.randomcutforest.testutils.ExampleDataSets.rotateClockWise;
import static java.lang.Math.PI;

import java.io.BufferedWriter;
import java.io.FileWriter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.returntypes.DensityOutput;

public class DynamicDensity implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicDensity().run();
    }

    @Override
    public String command() {
        return "dynamic_sampling";
    }

    @Override
    public String description() {
        return "check dynamic sampling";
    }

    /**
     * plot the dynamic_density_example using any tool in gnuplot one can plot the
     * directions to higher density via do for [i=0:358:2] {plot
     * "dynamic_density_example" index (i+1) u 1:2:3:4 w vectors t ""} or the raw
     * density at the points via do for [i=0:358:2] {plot "dynamic_density_example"
     * index i w p pt 7 palette t ""}
     * 
     * @throws Exception
     */
    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .build();
        String name = "/Users/sudipto/dynamic_density_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));
        double[][] data = generate(1000);
        double[] queryPoint;
        for (int degree = 0; degree < 360; degree += 2) {
            for (double[] datum : data) {
                newForest.update(rotateClockWise(datum, -2 * PI * degree / 360));
            }
            for (double[] datum : data) {
                queryPoint = rotateClockWise(datum, -2 * PI * degree / 360);
                DensityOutput density = newForest.getSimpleDensity(queryPoint);
                double value = density.getDensity(0.001, 2);
                file.append(queryPoint[0] + " " + queryPoint[1] + " " + value + "\n");
            }
            file.append("\n");
            file.append("\n");

            for (double x = -0.95; x < 1; x += 0.1) {
                for (double y = -0.95; y < 1; y += 0.1) {
                    DensityOutput density = newForest.getSimpleDensity(new double[] { x, y });
                    double aboveInY = density.getDirectionalDensity(0.001, 2).low[1];
                    double belowInY = density.getDirectionalDensity(0.001, 2).high[1];
                    double toTheLeft = density.getDirectionalDensity(0.001, 2).high[0];
                    double toTheRight = density.getDirectionalDensity(0.001, 2).low[0];
                    double len = Math.sqrt(aboveInY * aboveInY + belowInY * belowInY + toTheLeft * toTheLeft
                            + toTheRight * toTheRight);
                    file.append(x + " " + y + " " + ((toTheRight - toTheLeft) * 0.05 / len) + " "
                            + ((aboveInY - belowInY) * 0.05 / len) + "\n");
                }
            }
            file.append("\n");
            file.append("\n");
        }
        file.close();
    }
}
