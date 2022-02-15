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

package com.amazon.randomcutforest.examples.serialization;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class ObjectStreamExample implements Example {

    public static void main(String[] args) throws Exception {
        new ObjectStreamExample().run();
    }

    @Override
    public String command() {
        return "object_stream";
    }

    @Override
    public String description() {
        return "serialize a Random Cut Forest with object stream";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 10;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;

        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).build();

        int dataSize = 1000 * sampleSize;
        NormalMixtureTestData testData = new NormalMixtureTestData();
        for (double[] point : testData.generateTestData(dataSize, dimensions)) {
            forest.update(point);
        }

        // Convert to an array of bytes and print the size

        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
        System.out.printf("dimensions = %d, numberOfTrees = %d, sampleSize = %d, precision = %s%n", dimensions,
                numberOfTrees, sampleSize, precision);
        byte[] bytes = serialize(mapper.toState(forest));
        System.out.printf("Object output stream size = %d bytes%n", bytes.length);

        // Restore from object stream and compare anomaly scores produced by the two
        // forests

        RandomCutForestState state2 = (RandomCutForestState) deserialize(bytes);
        RandomCutForest forest2 = mapper.toModel(state2);

        int testSize = 100;
        double delta = Math.log(sampleSize) / Math.log(2) * 0.05;

        int differences = 0;
        int anomalies = 0;

        for (double[] point : testData.generateTestData(testSize, dimensions)) {
            double score = forest.getAnomalyScore(point);
            double score2 = forest2.getAnomalyScore(point);

            // we mostly care that points that are scored as an anomaly by one forest are
            // also scored as an anomaly by the other forest
            if (score > 1 || score2 > 1) {
                anomalies++;
                if (Math.abs(score - score2) > delta) {
                    differences++;
                }
            }

            forest.update(point);
            forest2.update(point);
        }

        // first validate that this was a nontrivial test
        if (anomalies == 0) {
            throw new IllegalStateException("test data did not produce any anomalies");
        }

        // validate that the two forests agree on anomaly scores
        if (differences >= 0.01 * testSize) {
            throw new IllegalStateException("restored forest does not agree with original forest");
        }

        System.out.println("Looks good!");
    }

    private byte[] serialize(Object model) {
        try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
             ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream)) {
            objectOutputStream.writeObject(model);
            objectOutputStream.flush();
            return byteArrayOutputStream.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException("Failed to serialize model.", e.getCause());
        }
    }

    private Object deserialize(byte[] modelBin) {
        try (ObjectInputStream objectInputStream = new ObjectInputStream(new ByteArrayInputStream(modelBin))) {
            return objectInputStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Failed to deserialize model.", e.getCause());
        }
    }
}
