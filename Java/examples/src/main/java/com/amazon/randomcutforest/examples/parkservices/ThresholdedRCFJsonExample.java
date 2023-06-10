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

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestState;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Serialize a Random Cut Forest to JSON using
 * <a href="https://github.com/FasterXML/jackson">Jackson</a>.
 */
public class ThresholdedRCFJsonExample implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedRCFJsonExample().run();
    }

    @Override
    public String command() {
        return "json";
    }

    @Override
    public String description() {
        return "serialize a Thresholded Random Cut Forest as a JSON string";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int baseDimension = 2;
        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        long seed = new Random().nextLong();
        System.out.println("seed :" + seed);
        Random rng = new Random(seed);

        int dimensions = baseDimension * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions)
                .shingleSize(shingleSize).transformMethod(TransformMethod.NORMALIZE).numberOfTrees(numberOfTrees)
                .sampleSize(sampleSize).build();

        int dataSize = 4 * sampleSize;
        int testSize = sampleSize;
        double[][] data = ShingledMultiDimDataWithKeys.getMultiDimData(dataSize + shingleSize - 1, 50, 100, 5,
                rng.nextLong(), baseDimension, 5.0, false).data;

        for (int i = 0; i < data.length - testSize; i++) {
            forest.process(data[i], 0L);
        }

        // Convert to JSON and print the number of bytes

        ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
        ObjectMapper jsonMapper = new ObjectMapper();

        String json = jsonMapper.writeValueAsString(mapper.toState(forest));

        System.out.printf("JSON size = %d bytes%n", json.getBytes().length);

        // Restore from JSON and compare anomaly scores produced by the two forests

        ThresholdedRandomCutForest forest2 = mapper
                .toModel(jsonMapper.readValue(json, ThresholdedRandomCutForestState.class));

        for (int i = data.length; i < data.length; i++) {
            AnomalyDescriptor result = forest.process(data[i], 0L);
            AnomalyDescriptor shadow = forest2.process(data[i], 0L);
            assert (Math.abs(result.getRCFScore() - shadow.getRCFScore()) < 1e-6);
        }

        System.out.println("Looks good!");
    }
}
