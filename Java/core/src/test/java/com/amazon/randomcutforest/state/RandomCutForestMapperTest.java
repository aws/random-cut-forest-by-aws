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
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.store.PointStoreFloat;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

public class RandomCutForestMapperTest {

    private static int dimensions = 5;
    private static int sampleSize = 128;

    private static Stream<RandomCutForest> compactForestProvider() {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions)
                .sampleSize(sampleSize);

        RandomCutForest cachedDouble = builder.boundingBoxCacheFraction(new Random().nextDouble())
                .precision(Precision.DOUBLE).build();
        RandomCutForest cachedFloat = builder.boundingBoxCacheFraction(new Random().nextDouble())
                .precision(Precision.SINGLE).build();
        RandomCutForest uncachedDouble = builder.boundingBoxCacheFraction(0.0).precision(Precision.DOUBLE).build();
        RandomCutForest uncachedFloat = builder.boundingBoxCacheFraction(0.0).precision(Precision.SINGLE).build();

        return Stream.of(cachedDouble, cachedFloat, uncachedDouble, uncachedFloat);
    }

    private RandomCutForestMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContext(true);
    }

    public void assertCompactForestEquals(RandomCutForest forest, RandomCutForest forest2) {
        assertEquals(forest.getDimensions(), forest2.getDimensions());
        assertEquals(forest.getSampleSize(), forest2.getSampleSize());
        assertEquals(forest.getOutputAfter(), forest2.getOutputAfter());
        assertEquals(forest.getNumberOfTrees(), forest2.getNumberOfTrees());
        assertEquals(forest.getLambda(), forest2.getLambda());
        assertEquals(forest.isStoreSequenceIndexesEnabled(), forest2.isStoreSequenceIndexesEnabled());
        assertEquals(forest.isCompactEnabled(), forest2.isCompactEnabled());
        assertEquals(forest.getPrecision(), forest2.getPrecision());
        assertEquals(forest.getBoundingBoxCacheFraction(), forest2.getBoundingBoxCacheFraction());
        assertEquals(forest.isCenterOfMassEnabled(), forest2.isCenterOfMassEnabled());
        assertEquals(forest.isParallelExecutionEnabled(), forest2.isParallelExecutionEnabled());
        assertEquals(forest.getThreadPoolSize(), forest2.getThreadPoolSize());

        PointStoreCoordinator coordinator = (PointStoreCoordinator) forest.getUpdateCoordinator();
        PointStoreCoordinator coordinator2 = (PointStoreCoordinator) forest2.getUpdateCoordinator();

        if (forest.getPrecision() == Precision.DOUBLE) {
            PointStoreDouble store = (PointStoreDouble) coordinator.getStore();
            PointStoreDouble store2 = (PointStoreDouble) coordinator2.getStore();
            assertArrayEquals(store.getRefCount(), store2.getRefCount());
            assertArrayEquals(store.getStore(), store2.getStore());
            assertEquals(store.getFreeIndexPointer(), store2.getFreeIndexPointer());
            assertEquals(store.getCapacity(), store2.getCapacity());
            assertEquals(store.size(), store2.size());

        } else {
            PointStoreFloat store = (PointStoreFloat) coordinator.getStore();
            PointStoreFloat store2 = (PointStoreFloat) coordinator2.getStore();
            assertArrayEquals(store.getRefCount(), store2.getRefCount());
            assertArrayEquals(store.getStore(), store2.getStore());
            assertEquals(store.getFreeIndexPointer(), store2.getFreeIndexPointer());
            assertEquals(store.getCapacity(), store2.getCapacity());
            assertEquals(store.size(), store2.size());

        }
    }

    @ParameterizedTest
    @MethodSource("compactForestProvider")
    public void testRoundTripForCompactForest(RandomCutForest forest) {

        NormalMixtureTestData testData = new NormalMixtureTestData();
        for (double[] point : testData.generateTestData(sampleSize, dimensions)) {
            forest.update(point);
        }

        RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));
        assertCompactForestEquals(forest, forest2);
    }

    @ParameterizedTest
    @MethodSource("compactForestProvider")
    public void testRoundTripForCompactForestSaveTreeState(RandomCutForest forest) {
        mapper.setSaveTreeState(true);
        testRoundTripForCompactForest(forest);
    }

    @Test
    public void testRoundTripForEmptyForest() {
        Precision precision = Precision.DOUBLE;

        RandomCutForest forest = RandomCutForest.builder().compactEnabled(true).dimensions(dimensions)
                .sampleSize(sampleSize).precision(precision).build();

        mapper.setSaveTreeState(true);
        RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));

        assertCompactForestEquals(forest, forest2);
    }
}
