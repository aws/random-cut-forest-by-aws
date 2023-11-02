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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.min;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import com.amazon.randomcutforest.PredictiveRandomCutForest;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.Summarizer;

public class ConditionalPredictive implements Example {

    public static void main(String[] args) throws Exception {
        new ConditionalPredictive().run();
    }

    @Override
    public String command() {
        return "Conditional_predictive_example";
    }

    @Override
    public String description() {
        return "An example that uses imputation for prediction";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int shingleSize = 1;
        int numberOfTrees = 100;
        int sampleSize = 256;
        int dataSize = 40 * sampleSize;

        // 5 dimensions, three are known and 4,5 th unknown (and stochastic)
        int baseDimensions = 5;

        PredictiveRandomCutForest forest = new PredictiveRandomCutForest.Builder<>().inputDimensions(baseDimensions)
                .randomSeed(0).numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
                .startNormalization(sampleSize / 2).transformMethod(TransformMethod.NORMALIZE).build();

        long seed = 17;
        new Random().nextLong();

        System.out.println("seed = " + seed);
        NormalDistribution normal = new NormalDistribution(new Random(seed));
        Random random = new Random(seed + 10);

        String name = "predictive_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));
        for (int i = 0; i < dataSize; i++) {
            float[] record = generateRecordKey(random);
            checkArgument(record[3] == 0, " should not be filled");
            checkArgument(record[4] == 0, " should not be filled");

            SampleSummary answer = forest.predict(record, 0, new int[] { 3, 4 });
            fillInValues(record, random, normal);
            forest.update(record, 0);
            double tag = Double.MAX_VALUE;
            for (int y = 0; y < answer.summaryPoints.length; y++) {
                double t = Summarizer.L2distance(record, answer.summaryPoints[y]);
                tag = min(tag, t);
            }

            file.append(record[0] + " " + record[1] + " " + record[2] + " " + record[3] + " " + record[4] + " " + tag
                    + "\n");
        }
        file.close();
    }

    float[] generateRecordKey(Random random) {
        float[] record = new float[5];
        double firstToss = random.nextDouble();
        double secondToss = random.nextDouble();
        double thirdToss = random.nextDouble();
        if (firstToss < 0.8) {
            record[0] = 1.0f;
            if (secondToss < 0.8) {
                record[1] = 19;
            } else {
                record[1] = 25;
            }
            record[2] = (float) thirdToss * 10;
        } else {
            record[0] = 0.0f;
            if (secondToss < 0.3) {
                record[1] = 16;
                record[2] = 12;
            } else {
                record[1] = 20;
                record[2] = 4;
            }
        }
        return record;
    }

    void fillInValues(float[] record, Random random, NormalDistribution normal) {
        if (record[0] < 0.5) {
            double next = random.nextDouble();
            record[3] = (float) ((next < 0.5) ? normal.nextDouble(20, 5) : normal.nextDouble(40, 5));
            record[4] = (float) normal.nextDouble(30, 3);
        } else {
            if (record[1] < 20) {
                record[3] = (float) normal.nextDouble(30, 10);
                record[4] = (float) normal.nextDouble(10, 3);
            } else {
                if (record[2] < 6) {
                    double next = random.nextDouble();
                    record[3] = (float) ((next < 0.3) ? normal.nextDouble(20, 5) : normal.nextDouble(40, 3));
                    record[4] = (float) normal.nextDouble(50, 1);
                } else {
                    double next = random.nextDouble();
                    record[3] = (float) normal.nextDouble(30, 1);
                    record[4] = (float) ((next < 0.7) ? normal.nextDouble(10, 3) : normal.nextDouble(30, 5));
                }
            }
        }
    }

    static class NormalDistribution {
        private final Random rng;
        private final double[] buffer;
        private int index;

        NormalDistribution(Random rng) {
            this.rng = rng;
            buffer = new double[2];
            index = 0;
        }

        double nextDouble() {
            if (index == 0) {
                // apply the Box-Muller transform to produce Normal variates
                double u = rng.nextDouble();
                double v = rng.nextDouble();
                double r = Math.sqrt(-2 * Math.log(u));
                buffer[0] = r * Math.cos(2 * Math.PI * v);
                buffer[1] = r * Math.sin(2 * Math.PI * v);
            }

            double result = buffer[index];
            index = (index + 1) % 2;

            return result;
        }

        double nextDouble(double mu, double sigma) {
            return mu + sigma * nextDouble();
        }
    }
}
