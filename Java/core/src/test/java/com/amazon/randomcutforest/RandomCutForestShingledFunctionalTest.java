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

package com.amazon.randomcutforest;

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys.generateShingledData;
import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.testutils.MultiDimDataWithKey;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.testutils.ShingledMultiDimDataWithKeys;
import com.amazon.randomcutforest.util.ShingleBuilder;

@Tag("functional")
public class RandomCutForestShingledFunctionalTest {
    private static int numberOfTrees;
    private static int sampleSize;
    private static int dimensions;
    private static int randomSeed;
    private static int shingleSize;
    private static ShingleBuilder shingleBuilder;
    private static RandomCutForest forest;

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    @BeforeAll
    public static void oneTimeSetUp() {
        numberOfTrees = 100;
        sampleSize = 256;
        dimensions = 2;
        randomSeed = 123;
        shingleSize = 3;

        shingleBuilder = new ShingleBuilder(dimensions, shingleSize);

        forest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shingleBuilder.getShingledPointSize()).randomSeed(randomSeed).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        dataSize = 10_000;

        baseMu = 0.0;
        baseSigma = 1.0;
        anomalyMu = 5.0;
        anomalySigma = 1.5;
        transitionToAnomalyProbability = 0.01;
        transitionToBaseProbability = 0.4;

        NormalMixtureTestData generator = new NormalMixtureTestData(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        double[][] data = generator.generateTestData(dataSize, dimensions);

        for (int i = 0; i < dataSize; i++) {
            shingleBuilder.addPoint(data[i]);
            if (shingleBuilder.isFull()) {
                forest.update(shingleBuilder.getShingle());
            }
        }
    }

    @Test
    public void testExtrapolateBasic() {
        double[] result = forest.extrapolateBasic(shingleBuilder.getShingle(), 4, dimensions, false);
        assertEquals(4 * dimensions, result.length);

        result = forest.extrapolateBasic(shingleBuilder.getShingle(), 4, dimensions, true, 2);
        assertEquals(4 * dimensions, result.length);

        result = forest.extrapolateBasic(shingleBuilder, 4);
        assertEquals(4 * dimensions, result.length);

        // use a block size which is too big
        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(shingleBuilder.getShingle(), 4, 4, true, 2));
    }

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    public void InternalShinglingTest(boolean rotation) {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 2;
        int dimensions = baseDimensions * shingleSize;
        long seed = new Random().nextLong();
        System.out.println(seed);

        int numTrials = 1; // test is exact equality, reducing the number of trials
        int length = 4000 * sampleSize;

        for (int i = 0; i < numTrials; i++) {

            RandomCutForest first = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(true)
                    .internalRotationEnabled(rotation).shingleSize(shingleSize).build();

            RandomCutForest second = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(false)
                    .shingleSize(shingleSize).build();

            RandomCutForest third = new RandomCutForest.Builder<>().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed).internalShinglingEnabled(false).shingleSize(1)
                    .build();

            MultiDimDataWithKey dataWithKeys = ShingledMultiDimDataWithKeys.getMultiDimData(length, 50, 100, 5,
                    seed + i, baseDimensions);

            double[][] shingledData = generateShingledData(dataWithKeys.data, shingleSize, baseDimensions, rotation);

            assertEquals(shingledData.length, dataWithKeys.data.length - shingleSize + 1);

            int count = shingleSize - 1;
            // insert initial points
            for (int j = 0; j < shingleSize - 1; j++) {
                first.update(dataWithKeys.data[j]);
            }

            for (int j = 0; j < shingledData.length; j++) {
                // validate equality of points
                for (int y = 0; y < baseDimensions; y++) {
                    int position = (rotation) ? (count % shingleSize) : shingleSize - 1;
                    assertEquals(dataWithKeys.data[count][y], shingledData[j][position * baseDimensions + y], 1e-10);
                }

                double firstResult = first.getAnomalyScore(dataWithKeys.data[count]);
                first.update(dataWithKeys.data[count]);
                ++count;
                double secondResult = second.getAnomalyScore(shingledData[j]);
                second.update(shingledData[j]);
                double thirdResult = third.getAnomalyScore(shingledData[j]);
                third.update(shingledData[j]);

                assertEquals(firstResult, secondResult, 1e-10);
                assertEquals(secondResult, thirdResult, 1e-10);
            }
            PointStore store = (PointStore) first.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
            store = (PointStore) second.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
            store = (PointStore) third.getUpdateCoordinator().getStore();
            assertEquals(store.getCurrentStoreCapacity() * dimensions, store.getStore().length);
        }
    }

    @Test
    public void testExtrapolateShingleAwareSinglePrecision() {

        int numberOfTrees = 100;
        int sampleSize = 256;
        int shinglesize = 10;
        long randomSeed = 123;

        RandomCutForest newforest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).shingleSize(shinglesize)
                .precision(Precision.FLOAT_32).build();
        RandomCutForest anotherforest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).shingleSize(1)
                .precision(Precision.FLOAT_32).build();
        RandomCutForest yetAnotherforest = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).shingleSize(shinglesize)
                .internalShinglingEnabled(true).precision(Precision.FLOAT_32).build();

        double amplitude = 50.0;
        double noise = 2.0;
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 850;
        double[] data = getDataA(amplitude, noise);
        double[] answer = null;
        double error = 0;
        double[] record = null;

        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            // input is always double[], internal representation is float[]
            // input is 1 dimensional for internal shingling (for 1 dimensional sequences)
            yetAnotherforest.update(new double[] { data[j] });

            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforest.update(record);
                anotherforest.update(record);
            }
        }

        answer = newforest.extrapolateBasic(record, 200, 1, false);
        double[] anotherAnswer = anotherforest.extrapolateBasic(record, 200, 1, false);
        double[] yetAnotherAnswer = yetAnotherforest.extrapolate(200);
        assertArrayEquals(anotherAnswer, answer, 1e-10);
        assertArrayEquals(yetAnotherAnswer, answer, 1e-10);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double prediction = amplitude * cos((j + 850 - 50) * 2 * PI / 120);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;

        assertTrue(error < 4 * noise);

    }

    @Test
    public void testExtrapolateInternalRotationSinglePrecision() {

        int numberOfTrees = 100;
        int sampleSize = 256;
        int shinglesize = 120;
        long randomSeed = 123;

        RandomCutForest newforestA = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).precision(Precision.FLOAT_32).build();

        RandomCutForest newforestB = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).internalShinglingEnabled(true)
                .internalRotationEnabled(true).compact(true).shingleSize(shinglesize).precision(Precision.FLOAT_32)
                .build();
        RandomCutForest newforestC = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).shingleSize(shinglesize)
                .precision(Precision.FLOAT_32).build();
        double amplitude = 50.0;
        double noise = 2.0;
        Random noiseprg = new Random(72);
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 850;
        double[] data = getDataA(amplitude, noise);
        double[] answer = null;
        double error = 0;

        double[] record = null;
        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }

            newforestB.update(new double[] { data[j] });
            if (filledShingleAtleastOnce) {
                // produce cyclic vectors
                record = getShinglePoint(history, 0, shinglesize);
                newforestA.update(record);
                newforestC.update(record);
            }
        }

        answer = newforestA.extrapolateBasic(record, 200, 1, true, entryIndex);
        double[] anotherAnswer = newforestB.extrapolate(200);
        double[] yetAnotherAnswer = newforestC.extrapolateBasic(record, 200, 1, true, entryIndex);
        assertArrayEquals(answer, yetAnotherAnswer, 1e-10);
        double[] othershingle = toDoubleArray(newforestB.lastShingledPoint());
        assertEquals(entryIndex, newforestB.nextSequenceIndex() % shinglesize);
        assertArrayEquals(record, othershingle, 1e-10);
        assertArrayEquals(answer, anotherAnswer, 1e-10);
        error = 0;
        for (int j = 0; j < 200; j++) {
            double prediction = amplitude * cos((j + 850 - 50) * 2 * PI / 120);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 4 * noise);

    }

    @Test
    public void testExtrapolateC() {

        int numberOfTrees = 100;
        int sampleSize = 256;
        int shinglesize = 20;
        long randomSeed = 124;

        // build two identical copies; we will be giving them different
        // subsequent inputs and test adaptation to stream evolution

        RandomCutForest newforestC = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).timeDecay(1.0 / 300).build();

        RandomCutForest newforestD = RandomCutForest.builder().numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .dimensions(shinglesize).randomSeed(randomSeed).compact(true).timeDecay(1.0 / 300).build();

        double amplitude = 50.0;
        double noise = 2.0;
        Random noiseprg = new Random(72);
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[shinglesize];
        int num = 1330;
        double[] data = getDataB(amplitude, noise);
        double[] answer = null;
        double error = 0;

        double[] record = null;
        for (int j = 0; j < num; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestC.update(record);
                newforestD.update(record);
            }
        }
        /**
         * the two forests are identical up to this point we will now provide two
         * different input to each num+2*expLife=1930, but since the shape of the
         * pattern remains the same in a phase shift, the prediction comes back to
         * "normal" fairly quickly.
         */

        for (int j = num; j < 1630; ++j) { // we stream here ....
            double t = cos(2 * PI * (j - 50) / 240);
            history[entryIndex] = amplitude * t + noise * noiseprg.nextDouble();
            ;
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestC.update(record);
            }
        }
        answer = newforestC.extrapolateBasic(record, 200, 1, false);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double t = cos(2 * PI * (1630 + j - 50) / 240);
            double prediction = amplitude * t;
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 2 * noise);

        /**
         * Here num+2*expLife=1930 for a small explife such as 300, num+expLife is
         * already sufficient increase the factor for larger expLife or increase the
         * sampleSize to absorb the longer range dependencies of a larger expLife
         */
        for (int j = num; j < 1630; ++j) { // we stream here ....
            double t = cos(2 * PI * (j + 50) / 120);
            int sign = (t > 0) ? 1 : -1;
            history[entryIndex] = amplitude * sign * Math.pow(t * sign, 1.0 / 3) + noise * noiseprg.nextDouble();
            entryIndex = (entryIndex + 1) % shinglesize;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {

                record = getShinglePoint(history, entryIndex, shinglesize);
                newforestD.update(record);
            }
        }
        answer = newforestD.extrapolateBasic(record, 200, 1, false);

        error = 0;
        for (int j = 0; j < 200; j++) {
            double t = cos(2 * PI * (1630 + j + 50) / 120);
            int sign = (t > 0) ? 1 : -1;
            double prediction = amplitude * sign * Math.pow(t * sign, 1.0 / 3);
            error += Math.abs(prediction - answer[j]);
        }
        error = error / 200;
        assertTrue(error < 2 * noise);
    }

    double[] getDataA(double amplitude, double noise) {
        int num = 850;
        double[] data = new double[num];
        Random noiseprg = new Random(9000);

        for (int i = 0; i < 510; i++) {
            data[i] = amplitude * cos(2 * PI * (i - 50) / 120) + noise * noiseprg.nextDouble();
        }
        for (int i = 510; i < 525; i++) { // flatline
            data[i] = 0;
        }
        for (int i = 525; i < 825; i++) {
            data[i] = amplitude * cos(2 * PI * (i - 50) / 120) + noise * noiseprg.nextDouble();
        }
        for (int i = 825; i < num; i++) { // high frequency noise
            data[i] = amplitude * cos(2 * PI * (i - 50) / 12) + noise * noiseprg.nextDouble();
        }
        return data;
    }

    double[] getDataB(double amplitude, double noise) {
        int num = 1330;
        double[] data = new double[num];
        Random noiseprg = new Random(9001);
        for (int i = 0; i < 990; i++) {
            data[i] = amplitude * cos(2 * PI * (i + 50) / 240) + noise * noiseprg.nextDouble();
        }
        for (int i = 990; i < 1005; i++) { // flatline
            data[i] = 0;
        }
        for (int i = 1005; i < 1305; i++) {
            data[i] = amplitude * cos(2 * PI * (i + 50) / 240) + noise * noiseprg.nextDouble();
        }
        for (int i = 1305; i < num; i++) { // high frequency noise
            data[i] = amplitude * cos(2 * PI * (i + 50) / 12) + noise * noiseprg.nextDouble();
        }
        return data;
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

    @Test
    public void testUpdate() {
        int dimensions = 10;

        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(100).compact(true).dimensions(dimensions)
                .randomSeed(0).sampleSize(200).precision(Precision.FLOAT_32).build();

        double[][] trainingData = genShingledData(1000, dimensions, 0);
        double[][] testData = genShingledData(100, dimensions, 1);

        for (int i = 0; i < testData.length; i++) {

            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContextEnabled(true);
            mapper.setSaveTreeStateEnabled(true);

            double score = forest.getAnomalyScore(testData[i]);
            forest.update(testData[i]);
            RandomCutForestState forestState = mapper.toState(forest);
            forest = mapper.toModel(forestState);
        }
    }

    private static double[][] genShingledData(int size, int dimensions, long seed) {
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

    private static double[] getDataD(int num, double amplitude, double noise, long seed) {

        double[] data = new double[num];
        Random noiseprg = new Random(seed);
        for (int i = 0; i < num; i++) {
            data[i] = amplitude * cos(2 * PI * (i + 50) / 1000) + noise * noiseprg.nextDouble();
        }

        return data;
    }
}
