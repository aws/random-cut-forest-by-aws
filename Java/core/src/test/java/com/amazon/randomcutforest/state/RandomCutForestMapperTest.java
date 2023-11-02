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

package com.amazon.randomcutforest.state;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.preprocessor.IPreprocessor;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorMapper;
import com.amazon.randomcutforest.state.preprocessor.PreprocessorState;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

public class RandomCutForestMapperTest {

    private static int dimensions = 5;
    private static int sampleSize = 128;

    private Version version = new Version();

    private static Stream<RandomCutForest> compactForestProvider() {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true).dimensions(dimensions)
                .sampleSize(sampleSize);

        RandomCutForest cachedFloat = builder.boundingBoxCacheFraction(new Random().nextDouble()).build();
        RandomCutForest uncachedFloat = builder.boundingBoxCacheFraction(0.0).build();

        return Stream.of(cachedFloat, uncachedFloat);
    }

    private RandomCutForestMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
    }

    public void assertCompactForestEquals(RandomCutForest forest, RandomCutForest forest2, boolean saveTree) {
        assertEquals(forest.getDimensions(), forest2.getDimensions());
        assertEquals(forest.getSampleSize(), forest2.getSampleSize());
        assertEquals(forest.getOutputAfter(), forest2.getOutputAfter());
        assertEquals(forest.getNumberOfTrees(), forest2.getNumberOfTrees());
        assertEquals(forest.getTimeDecay(), forest2.getTimeDecay());
        assertEquals(forest.isStoreSequenceIndexesEnabled(), forest2.isStoreSequenceIndexesEnabled());
        assertEquals(forest.isCompact(), forest2.isCompact());
        assertEquals(forest.getPrecision(), forest2.getPrecision());
        assertEquals(forest.getBoundingBoxCacheFraction(), forest2.getBoundingBoxCacheFraction());
        assertEquals(forest.isCenterOfMassEnabled(), forest2.isCenterOfMassEnabled());
        assertEquals(forest.isParallelExecutionEnabled(), forest2.isParallelExecutionEnabled());
        assertEquals(forest.getThreadPoolSize(), forest2.getThreadPoolSize());

        PointStoreCoordinator coordinator = (PointStoreCoordinator) forest.getUpdateCoordinator();
        PointStoreCoordinator coordinator2 = (PointStoreCoordinator) forest2.getUpdateCoordinator();

        PointStore store = (PointStore) coordinator.getStore();
        PointStore store2 = (PointStore) coordinator2.getStore();
        assertArrayEquals(store.getRefCount(), store2.getRefCount());
        assertArrayEquals(store.getStore(), store2.getStore());
        assertEquals(store.getCapacity(), store2.getCapacity());
        assertEquals(store.size(), store2.size());

        ComponentList<?, ?> components = forest.getComponents();
        ComponentList<?, ?> otherComponents = new ComponentList(forest2.getComponents());
        for (int i = 0; i < components.size(); i++) {
            SamplerPlusTree first = (SamplerPlusTree<?, ?>) components.get(i);
            SamplerPlusTree second = (SamplerPlusTree<?, ?>) otherComponents.get(i);
            if (saveTree) {
                assertEquals(first.getTree().getRandomSeed(), second.getTree().getRandomSeed());
            }
            assertEquals(((CompactSampler) first.getSampler()).getRandomSeed(),
                    ((CompactSampler) second.getSampler()).getRandomSeed());
        }
    }

    void testForest(RandomCutForest forest, Boolean saveTree) {
        NormalMixtureTestData testData = new NormalMixtureTestData();
        for (double[] point : testData.generateTestData(sampleSize, dimensions)) {
            forest.update(point);
        }
        RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));
        assertCompactForestEquals(forest, forest2, saveTree);

    }

    @ParameterizedTest
    @MethodSource("compactForestProvider")
    public void testRoundTripForCompactForest(RandomCutForest forest) {
        testForest(forest, false);
    }

    @ParameterizedTest
    @MethodSource("compactForestProvider")
    public void testRoundTripForCompactForestSaveTreeState(RandomCutForest forest) {
        mapper.setSaveTreeStateEnabled(true);
        testForest(forest, true);
    }

    @ParameterizedTest
    @MethodSource("compactForestProvider")
    public void testRoundTripForCompactForestSaveTreeStatePartial(RandomCutForest forest) {
        mapper.setSaveTreeStateEnabled(true);
        mapper.setPartialTreeStateEnabled(true);
        testRoundTripForCompactForest(forest);
    }

    @Test
    void testSaveSamplers() {
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).sampleSize(sampleSize)
                .numberOfTrees(1).build();
        NormalMixtureTestData testData = new NormalMixtureTestData();
        for (double[] point : testData.generateTestData(sampleSize, dimensions)) {
            forest.update(point);
        }
        mapper.setSaveSamplerStateEnabled(false);
        assertThrows(IllegalArgumentException.class, () -> mapper.toModel(mapper.toState(forest), 10));
        mapper.setSaveSamplerStateEnabled(true);
    }

    @Test
    void executionContext() {
        ExecutionContext ec = new ExecutionContext();
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).sampleSize(sampleSize)
                .parallelExecutionEnabled(true).threadPoolSize(23).numberOfTrees(1).build();
        RandomCutForest forest2 = mapper.toModel(mapper.toState(forest), ec);
        assertFalse(forest2.isParallelExecutionEnabled());
        assertEquals(0, forest2.getThreadPoolSize());
    }

    @Test
    void testVersion() {
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).sampleSize(sampleSize)
                .parallelExecutionEnabled(true).threadPoolSize(23).numberOfTrees(1).build();
        assertEquals(mapper.toState(forest).getVersion(), version.V4_0);
    }

    @Test
    void testPrecisionException() {
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).sampleSize(sampleSize)
                .parallelExecutionEnabled(true).threadPoolSize(23).numberOfTrees(1).build();
        RandomCutForestState state = mapper.toState(forest);
        assertDoesNotThrow(() -> mapper.toModel(state, 0L));
        state.setPrecision(Precision.FLOAT_64.name());
        assertThrows(IllegalStateException.class, () -> mapper.toModel(state, 0));
    }

    @Test
    public void testRoundTripForEmptyForest() {
        Precision precision = Precision.FLOAT_64;

        RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).sampleSize(sampleSize)
                .numberOfTrees(1).build();

        mapper.setSaveTreeStateEnabled(true);
        RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));

        assertCompactForestEquals(forest, forest2, true);
    }

    @Test
    public void testRoundTripForSingleNodeForest() {
        int dimensions = 10;
        long seed = new Random().nextLong();
        System.out.println(" Seed " + seed);
        RandomCutForest forest = RandomCutForest.builder().compact(true).dimensions(dimensions).numberOfTrees(1)
                .precision(Precision.FLOAT_32).internalShinglingEnabled(false).randomSeed(seed).build();
        Random r = new Random(seed + 1);
        double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
        for (int i = 0; i < new Random().nextInt(1000); i++) {
            forest.update(point);
        }
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
        mapper.setSaveTreeStateEnabled(true);
        mapper.setPartialTreeStateEnabled(true);
        RandomCutForest copyForest = mapper.toModel(mapper.toState(forest));

        for (int i = 0; i < new Random(seed + 2).nextInt(1000); i++) {
            double[] anotherPoint = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
            assertEquals(forest.getAnomalyScore(anotherPoint), copyForest.getAnomalyScore(anotherPoint), 1e-10);
            forest.update(anotherPoint);
            copyForest.update(anotherPoint);
        }
    }

    private static float[] generate(int input) {
        return new float[] { (float) (20 * Math.sin(input / 10.0)), (float) (20 * Math.cos(input / 10.0)) };
    }

    @Test
    void benchmarkMappers() {
        long seed = new Random().nextLong();
        System.out.println(" Seed " + seed);
        Random random = new Random(seed);

        RandomCutForest rcf = RandomCutForest.builder().dimensions(2 * 10).shingleSize(10).sampleSize(628)
                .internalShinglingEnabled(true).randomSeed(random.nextLong()).build();
        for (int i = 0; i < 10000; i++) {
            rcf.update(generate(i));
        }
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
        mapper.setSaveTreeStateEnabled(true);
        for (int j = 0; j < 1000; j++) {
            RandomCutForest newRCF = mapper.toModel(mapper.toState(rcf));
            float[] test = generate(10000 + j);
            assertEquals(newRCF.getAnomalyScore(test), rcf.getAnomalyScore(test), 1e-6);
            rcf.update(test);
        }
    }

    @ParameterizedTest
    @EnumSource(V2RCFJsonResource.class)
    public void testJson(V2RCFJsonResource jsonResource) throws JsonProcessingException {
        RandomCutForestMapper rcfMapper = new RandomCutForestMapper();
        String json = getStateFromFile(jsonResource.getResource());
        assertNotNull(json);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        RandomCutForestState state = mapper.readValue(json, RandomCutForestState.class);
        RandomCutForest forest = rcfMapper.toModel(state);
        Random r = new Random(0);
        for (int i = 0; i < 20000; i++) {
            double[] point = r.ints(forest.getDimensions(), 0, 50).asDoubleStream().toArray();
            forest.getAnomalyScore(point);
            forest.update(point, 0L);
        }
        assertNotNull(forest);
    }

    @ParameterizedTest
    @EnumSource(V2PreProcessorJsonResource.class)
    public void testPreprocessorJson(V2PreProcessorJsonResource jsonResource) throws JsonProcessingException {
        PreprocessorMapper preMapper = new PreprocessorMapper();
        String json = getStateFromFile(jsonResource.getResource());
        assertNotNull(json);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        PreprocessorState state = mapper.readValue(json, PreprocessorState.class);
        IPreprocessor preprocessor = preMapper.toModel(state);
        assertNotNull(preprocessor);
    }

    private String getStateFromFile(String resourceFile) {
        try (InputStream is = RandomCutForestMapperTest.class.getResourceAsStream(resourceFile);
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }
            return b.toString();
        } catch (IOException e) {
            fail("Unable to load resource");
        }
        return null;
    }
}
