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

import static java.lang.Math.PI;

import java.util.Random;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;

import io.protostuff.LinkedBuffer;
import io.protostuff.ProtostuffIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;

/**
 * Serialize a Random Cut Forest using the
 * <a href="https://github.com/protostuff/protostuff">protostuff</a> library.
 */
public class ProtostuffExampleWithShingles implements Example {
    public static void main(String[] args) throws Exception {
        new ProtostuffExampleWithShingles().run();
    }

    @Override
    public String command() {
        return "protostuffWithShingles";
    }

    @Override
    public String description() {
        return "serialize a Random Cut Forest with the protostuff library for shingled points";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 10;
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_64;
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).precision(precision).shingleSize(dimensions)
                .build();
        int count = 1;
        int dataSize = 1000 * sampleSize;
        for (double[] point : generateShingledData(dataSize, dimensions, 0)) {
            forest.update(point);
        }

        // Convert to an array of bytes and print the size

        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
        mapper.setSaveTreeStateEnabled(false);

        Schema<RandomCutForestState> schema = RuntimeSchema.getSchema(RandomCutForestState.class);
        LinkedBuffer buffer = LinkedBuffer.allocate(512);
        byte[] bytes;
        try {
            RandomCutForestState state = mapper.toState(forest);
            bytes = ProtostuffIOUtil.toByteArray(state, schema, buffer);
        } finally {
            buffer.clear();
        }

        System.out.printf("dimensions = %d, numberOfTrees = %d, sampleSize = %d, precision = %s%n", dimensions,
                numberOfTrees, sampleSize, precision);
        System.out.printf("protostuff size = %d bytes%n", bytes.length);

        // Restore from protostuff and compare anomaly scores produced by the two
        // forests

        RandomCutForestState state2 = schema.newMessage();
        ProtostuffIOUtil.mergeFrom(bytes, state2, schema);
        RandomCutForest forest2 = mapper.toModel(state2);

        int testSize = 10000;
        double delta = Math.log(sampleSize) / Math.log(2) * 0.05;

        int differences = 0;
        int anomalies = 0;

        for (double[] point : generateShingledData(testSize, dimensions, 2)) {
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

        // validate that the two forests agree on anomaly scores
        if (differences >= 0.01 * testSize) {
            throw new IllegalStateException("restored forest does not agree with original forest");
        }

        System.out.println("Looks good!");
    }

    private double[][] generateShingledData(int size, int dimensions, long seed) {
        double[][] answer = new double[size][];
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[dimensions];
        int count = 0;
        double[] data = getDataD(size + dimensions - 1, 100, 5, seed);
        for (int j = 0; j < size + dimensions - 1; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % dimensions;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                // System.out.println("Adding " + j);
                answer[count++] = getShinglePoint(history, entryIndex, dimensions);
            }
        }
        return answer;
    }

    private static double[] getShinglePoint(double[] recentPointsSeen, int indexOfOldestPoint, int shingleLength) {
        double[] shingledPoint = new double[shingleLength];
        int i = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            shingledPoint[i++] = point;

        }
        return shingledPoint;
    }

    double[] getDataD(int num, double amplitude, double noise, long seed) {

        double[] data = new double[num];
        Random noiseprg = new Random(seed);
        for (int i = 0; i < num; i++) {
            data[i] = amplitude * Math.cos(2 * PI * (i + 50) / 1000) + noise * noiseprg.nextDouble();
        }

        return data;
    }
}
