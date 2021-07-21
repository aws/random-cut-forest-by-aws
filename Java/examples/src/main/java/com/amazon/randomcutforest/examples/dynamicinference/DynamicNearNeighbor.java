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

package com.amazon.randomcutforest.examples.dynamicinference;

import static com.amazon.randomcutforest.testutils.ExampleDataSets.generate;
import static com.amazon.randomcutforest.testutils.ExampleDataSets.rotateClockWise;
import static java.lang.Math.PI;

import java.io.BufferedWriter;
import java.io.FileWriter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.examples.Example;

public class DynamicNearNeighbor implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicNearNeighbor().run();
    }

    @Override
    public String command() {
        return "dynamic_near_neighbor";
    }

    @Override
    public String description() {
        return "shows an example of dynamic near neighbor computation where both the data and query are "
                + "evolving in time";
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        String name = "dynamic_near_neighbor_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));
        double[][] data = generate(1000);
        double[] queryPoint = new double[] { 0.5, 0.6 };
        for (int degree = 0; degree < 360; degree += 2) {
            for (double[] datum : data) {
                double[] transformed = rotateClockWise(datum, -2 * PI * degree / 360);
                file.append(transformed[0] + " " + transformed[1] + "\n");
                newForest.update(transformed);
            }
            file.append("\n");
            file.append("\n");

            double[] movingQuery = rotateClockWise(queryPoint, -3 * PI * degree / 360);
            double[] neighbor = newForest.getNearNeighborsInSample(movingQuery, 1).get(0).point;
            file.append(movingQuery[0] + " " + movingQuery[1] + " " + (neighbor[0] - movingQuery[0]) + " "
                    + (neighbor[1] - movingQuery[1]) + "\n");
            file.append("\n");
            file.append("\n");
        }
    }
}
