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

package com.amazon.randomcutforest.store;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PointStoreTest {

    private int dimensions;
    private int capacity;
    private PointStore pointStore;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        capacity = 4;
        pointStore = new PointStoreSmall(dimensions, capacity);
    }

    @Test
    public void testNew() {
        assertEquals(dimensions, pointStore.getDimensions());
        assertEquals(capacity, pointStore.getCapacity());
        assertEquals(0, pointStore.size());

        for (int i = 0; i < pointStore.getIndexCapacity(); i++) {
            assertEquals(0, pointStore.getRefCount(i));
        }
        pointStore.add(new float[2], 0);
        int index = pointStore.add(new float[2], 0);
        assertEquals(index, 1);
        for (int y = 0; y < 1000; y++) {
            pointStore.incrementRefCount(index);
        }
        assertEquals(pointStore.getRefCount(index), 1001);
        int[] counts = pointStore.getRefCount();
        assertEquals(counts[0], 1);
        assertEquals(counts[index], 1001);
        assertThrows(AssertionError.class, () -> pointStore.setLocation(0, 13));
        assertThrows(AssertionError.class, () -> pointStore.extendLocationList(-10));
    }

    @Test
    public void testConstructors() {
        PointStore.Builder builder = new PointStore.Builder().dynamicResizingEnabled(true);
        assertThrows(IllegalArgumentException.class, () -> new PointStoreSmall(builder));
        builder.dimensions(1000);
        assertThrows(IllegalArgumentException.class, () -> new PointStoreSmall(builder));
        builder.capacity(100000);
        assertThrows(IllegalArgumentException.class, () -> new PointStoreSmall(builder));
        assertDoesNotThrow(() -> new PointStoreLarge(builder));
        builder.shingleSize(3);
        assertThrows(IllegalArgumentException.class, () -> new PointStoreLarge(builder));
        builder.shingleSize(1);
        builder.dimensions(2);
        PointStoreLarge large = new PointStoreLarge(builder);
        assertThrows(IllegalArgumentException.class, () -> large.checkFeasible(0));
        assertEquals(large.size(), 0);
        large.add(new float[2], 0L);
        assertEquals(large.size(), 1);
    }

    @Test
    public void testAdd() {
        float[] point1 = { 1.2f, -3.4f };
        int offset1 = pointStore.add(point1, 1);
        assertTrue(offset1 >= 0 && offset1 < capacity);
        assertEquals(1, pointStore.getRefCount(offset1));
        assertEquals(1, pointStore.size());

        float[] retrievedPoint1 = pointStore.getNumericVector(offset1);
        assertNotSame(point1, retrievedPoint1);
        assertArrayEquals(point1, retrievedPoint1);

        float[] point2 = { 111.2f, -333.4f };
        int offset2 = pointStore.add(point2, 2);
        assertTrue(offset2 >= 0 && offset2 < capacity);
        assertEquals(1, pointStore.getRefCount(offset2));
        assertEquals(2, pointStore.size());
        assertNotEquals(offset1, offset2);

        float[] retrievedPoint2 = pointStore.getNumericVector(offset2);
        assertNotSame(point2, retrievedPoint2);
        assertArrayEquals(point2, retrievedPoint2);

        // check that adding a second point didn't change the first stored point's value
        retrievedPoint1 = pointStore.getNumericVector(offset1);
        assertNotSame(point1, retrievedPoint1);
        assertArrayEquals(point1, retrievedPoint1);
    }

    @Test
    public void testAddInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.add(new float[] { 1.1f, -2.2f, 3.0f }, 0));
        for (int i = 0; i < capacity; i++) {
            float[] point = new float[dimensions];
            point[0] = (float) Math.random();
            point[1] = (float) Math.random();
            pointStore.add(point, i + 2);
        }
        // point store is full
        assertThrows(IllegalStateException.class, () -> pointStore.add(new float[] { 1.1f, -2.2f }, 0));
    }

    @Test
    public void testGetInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.getNumericVector(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.getNumericVector(capacity));
    }

    @Test
    public void testIncrementRefCount() {
        float[] point = { 1.2f, -3.4f };
        int offset = pointStore.add(point, 0);
        assertEquals(1, pointStore.getRefCount(offset));

        pointStore.incrementRefCount(offset);
        assertEquals(2, pointStore.getRefCount(offset));
    }

    @Test
    public void testIncrementRefCountInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.incrementRefCount(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.incrementRefCount(0));
    }

    @Test
    public void testDecrementRefCount() {
        float[] point = { 1.2f, -3.4f };
        int offset = pointStore.add(point, 0);
        pointStore.incrementRefCount(offset);
        assertEquals(2, pointStore.getRefCount(offset));
        assertEquals(1, pointStore.size());

        pointStore.decrementRefCount(offset);
        assertEquals(1, pointStore.getRefCount(offset));
        assertEquals(1, pointStore.size());

        pointStore.decrementRefCount(offset);
        assertEquals(0, pointStore.getRefCount(offset));
        assertEquals(0, pointStore.size());
    }

    @Test
    public void testDecrementRefCountInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.decrementRefCount(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.decrementRefCount(0));
    }

    @Test
    public void testPointEquals() {
        float[] point = { 1.2f, -3.4f };
        int offset = pointStore.add(point, 0);
        assertArrayEquals(pointStore.getNumericVector(offset), point);
        assertNotEquals(pointStore.getNumericVector(offset), new float[] { 5.6f, -7.8f });
    }

    @Test
    public void testPointEqualsInvalid() {
        float[] point = { 1.2f, -3.4f };
        assertThrows(IllegalArgumentException.class, () -> pointStore.getNumericVector(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.getNumericVector(0));
    }

    @Test
    public void internalShinglingTestNoRotation() {
        int shinglesize = 10;
        PointStore store = new PointStore.Builder().capacity(20 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .currentStoreCapacity(1).build();
        assertFalse(store.isInternalRotationEnabled());
        Random random = new Random(0);
        float[] shingle = new float[shinglesize];
        for (int i = 0; i < 10 * shinglesize - 3; i++) {
            shingle[(i + 3) % shinglesize] = (float) random.nextDouble();
            store.add(new float[] { shingle[(i + 3) % shinglesize] }, i);
        }
        assertArrayEquals(store.getNumericVector(9 * shinglesize - 3), shingle, (float) 1e-6);
        assertArrayEquals(store.getInternalShingle(), shingle, (float) 1e-6);
        assertArrayEquals(store.transformIndices(new int[] { 0 }), new int[] { shinglesize - 1 });
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 1 }));
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 0, 0 }));
        assertArrayEquals(store.transformToShingledPoint(new float[] { 0.0f }),
                store.transformToShingledPoint(new float[] { -0.0f }), (float) 1e-6);
        assertThrows(IllegalArgumentException.class, () -> store.add(new float[] { 0, 0 }, 0));
    }

    @Test
    public void internalShinglingTestWithRotation() {
        int shinglesize = 10;
        PointStore store = new PointStore.Builder().capacity(20 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .internalRotationEnabled(true).currentStoreCapacity(1).build();
        assertTrue(store.isInternalRotationEnabled());
        Random random = new Random(0);
        float[] shingle = new float[shinglesize];
        float[] temp = null;
        for (int i = 0; i < 10 * shinglesize + 5; i++) {
            shingle[i % shinglesize] = (float) random.nextDouble();
            temp = store.transformToShingledPoint(new float[] { shingle[i % shinglesize] });
            store.add(new float[] { shingle[i % shinglesize] }, i);
        }
        assertEquals(store.getNextSequenceIndex(), 10 * shinglesize + 5);
        assertArrayEquals(temp, shingle, (float) 1e-6);
        assertArrayEquals(store.getNumericVector(9 * shinglesize + 5), shingle, (float) 1e-6);
        assertNotEquals(store.internalShingle, store.getInternalShingle());
        assertArrayEquals(store.getNumericVector(9 * shinglesize + 5), shingle);
        assertNotEquals(store.getNumericVector(9 * shinglesize + 4), shingle);
        assertArrayEquals(store.getInternalShingle(), shingle, (float) 1e-6);
        assertArrayEquals(store.transformIndices(new int[] { 0 }), new int[] { 5 });
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 1 }));
        assertEquals(store.transformToShingledPoint(new float[] { 1, 2 }).length, 2);
        assertArrayEquals(store.transformToShingledPoint(new float[] { 0.0f }),
                store.transformToShingledPoint(new float[] { -0.0f }), (float) 1e-6);
    }

    @Test
    public void checkRotationAndCompact() {
        int shinglesize = 4;
        PointStore store = new PointStore.Builder().capacity(2 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .internalRotationEnabled(true).currentStoreCapacity(1).build();
        for (int i = 0; i < 2 * shinglesize; i++) {
            store.add(new float[] { -i - 1 }, i);
        }
        for (int i = 0; i < 2 * shinglesize - shinglesize + 1; i++) {
            if (i != shinglesize - 1) {
                store.decrementRefCount(i);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> store.getNumericVector(0));
        float[] test = new float[shinglesize];
        for (int i = 0; i < shinglesize; i++) {
            test[i] = -(i + shinglesize + 1);
        }
        test[shinglesize - 1] = -shinglesize;
        assertArrayEquals(store.getNumericVector(shinglesize - 1), test, 1e-6f);
        store.compact();
        for (int i = 2 * shinglesize; i < 4 * shinglesize - 1; i++) {
            store.add(new float[] { -i - 1 }, i);
        }
        assertThrows(IllegalStateException.class, () -> store.add(new float[] { -4 * shinglesize }, 0));
        for (int i = 0; i < 2 * shinglesize; i++) {
            if (i != shinglesize - 1) {
                store.decrementRefCount(i);
            }
        }
        assertEquals(store.toString(shinglesize - 1), Arrays.toString(test));
        for (int i = 4 * shinglesize; i < 6 * shinglesize - 1; i++) {
            store.add(new float[] { -i - 1 }, i);
        }
        assertThrows(IllegalStateException.class,
                () -> store.add(new float[] { -6 * shinglesize }, 6 * shinglesize - 1));
        store.decrementRefCount(shinglesize - 1);
        store.add(new float[] { -6 * shinglesize }, 6 * shinglesize - 1);
        store.decrementRefCount(shinglesize);
        store.compact();
    }

    @Test
    void CompactionTest() {
        int shinglesize = 2;
        PointStore store = new PointStore.Builder().capacity(6).dimensions(shinglesize).shingleSize(shinglesize)
                .indexCapacity(6).directLocationEnabled(false).internalShinglingEnabled(true).build();

        store.add(new float[] { 0 }, 0L);
        for (int i = 0; i < 5; i++) {
            store.add(new float[] { i + 1 }, 0L);
        }
        int finalIndex = store.add(new float[] { 4 + 2 }, 0L);
        assertArrayEquals(store.getNumericVector(finalIndex), new float[] { 5, 6 });
        store.decrementRefCount(1);
        store.decrementRefCount(2);
        int index = store.add(new float[] { 7 }, 0L);
        assertArrayEquals(store.getNumericVector(index), new float[] { 6, 7 });
        store.decrementRefCount(index);
        assertTrue(store.size() < store.capacity);
        index = store.add(new float[] { 8 }, 0L);
        assertArrayEquals(store.getNumericVector(index), new float[] { 7, 8 });
    }

    @Test
    public void indexIntervalTest() {
        assertThrows(IllegalArgumentException.class, () -> new IndexIntervalManager(0));
        assertThrows(IllegalArgumentException.class, () -> new IndexIntervalManager(1, 0, null));
        assertThrows(IllegalArgumentException.class, () -> IndexIntervalManager.toBits(null));
        IndexIntervalManager a = new IndexIntervalManager(new int[] { 0, 1 }, 2);
        IndexIntervalManager manager = new IndexIntervalManager(1);
        manager.takeIndex();
        assertThrows(IllegalStateException.class, () -> manager.takeIndex());
        assertThrows(IllegalArgumentException.class, () -> manager.extendCapacity(1));
        manager.extendCapacity(2);
        manager.extendCapacity(3);
        assertEquals(manager.getCapacity(), 3);
    }
}
