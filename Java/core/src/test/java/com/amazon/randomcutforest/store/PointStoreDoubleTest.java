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

public class PointStoreDoubleTest {

    private int dimensions;
    private int capacity;
    private PointStoreDouble pointStore;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        capacity = 4;
        pointStore = new PointStoreDouble(dimensions, capacity);
    }

    @Test
    public void testNew() {
        assertEquals(dimensions, pointStore.getDimensions());
        assertEquals(capacity, pointStore.getCapacity());
        assertEquals(0, pointStore.size());

        for (int i = 0; i < pointStore.getIndexCapacity(); i++) {
            assertEquals(0, pointStore.getRefCount(i));
        }
    }

    @Test
    public void testAdd() {
        double[] point1 = { 1.2, -3.4 };
        int offset1 = pointStore.add(point1, 1);
        assertTrue(offset1 >= 0 && offset1 < capacity);
        assertEquals(1, pointStore.getRefCount(offset1));
        assertEquals(1, pointStore.size());

        double[] retrievedPoint1 = pointStore.get(offset1);
        assertNotSame(point1, retrievedPoint1);
        assertArrayEquals(point1, retrievedPoint1);

        double[] point2 = { 111.2, -333.4 };
        int offset2 = pointStore.add(point2, 2);
        assertTrue(offset2 >= 0 && offset2 < capacity);
        assertEquals(1, pointStore.getRefCount(offset2));
        assertEquals(2, pointStore.size());
        assertNotEquals(offset1, offset2);

        double[] retrievedPoint2 = pointStore.get(offset2);
        assertNotSame(point2, retrievedPoint2);
        assertArrayEquals(point2, retrievedPoint2);

        // check that adding a second point didn't change the first stored point's value
        retrievedPoint1 = pointStore.get(offset1);
        assertNotSame(point1, retrievedPoint1);
        assertArrayEquals(point1, retrievedPoint1);
    }

    @Test
    public void testAddInvalid() {
        // invalid dimensions in point
        assertThrows(IllegalArgumentException.class, () -> pointStore.add(new double[] { 1.1, -2.2, 3.3 }, 3));

        for (int i = 0; i < capacity; i++) {
            double[] point = new double[dimensions];
            point[0] = (float) Math.random();
            point[1] = (float) Math.random();
            pointStore.add(point, i + 2);
        }
        // point store is full
        assertThrows(IllegalStateException.class, () -> pointStore.add(new double[] { 1.1, -2.2 }, 0));
    }

    @Test
    public void testGetInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.get(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.get(capacity));
    }

    @Test
    public void testIncrementRefCount() {
        double[] point = { 1.2, -3.4 };
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
        double[] point = { 1.2, -3.4 };
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
        double[] point = { 1.2, -3.4 };
        int offset = pointStore.add(point, 0);
        assertTrue(pointStore.pointEquals(offset, point));
        assertFalse(pointStore.pointEquals(offset, new double[] { 5.6, -7.8 }));
    }

    @Test
    public void testPointEqualsInvalid() {
        double[] point = { 1.2, -3.4 };
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(-1, point));
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(0, point));

        int offset = pointStore.add(point, 0);
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(offset, new double[] { 99.9 }));
    }

    @Test
    public void internalshinglingTestNoRotation() {
        int shinglesize = 10;
        PointStoreDouble store = new PointStoreDouble.Builder().capacity(20 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .currentStoreCapacity(1).build();
        assertFalse(store.isInternalRotationEnabled());
        Random random = new Random(0);
        double[] shingle = new double[shinglesize];
        for (int i = 0; i < 10 * shinglesize - 3; i++) {
            shingle[(i + 3) % shinglesize] = random.nextDouble();
            store.add(new double[] { shingle[(i + 3) % shinglesize] }, i);
        }
        assertArrayEquals(store.get(9 * shinglesize - 3), shingle, 1e-10);
        assertArrayEquals(store.getInternalShingle(), shingle, 1e-10);
        assertTrue(store.pointEquals(9 * shinglesize - 3, shingle));
        assertArrayEquals(store.transformIndices(new int[] { 0 }), new int[] { shinglesize - 1 });
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 1 }));
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 0, 0 }));
        assertArrayEquals(store.transformToShingledPoint(new double[] { 0.0 }),
                store.transformToShingledPoint(new double[] { -0.0 }), 1e-6);
        assertThrows(IllegalArgumentException.class, () -> store.add(new double[] { 0, 0 }, 0));
    }

    @Test
    public void internalshinglingTestWithRotation() {
        int shinglesize = 10;
        PointStoreDouble store = new PointStoreDouble.Builder().capacity(20 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .internalRotationEnabled(true).currentStoreCapacity(1).build();
        assertTrue(store.isInternalRotationEnabled());
        Random random = new Random(0);
        double[] shingle = new double[shinglesize];
        double[] temp = null;
        for (int i = 0; i < 10 * shinglesize + 5; i++) {
            shingle[i % shinglesize] = random.nextDouble();
            temp = store.transformToShingledPoint(new double[] { shingle[i % shinglesize] });
            store.add(new double[] { shingle[i % shinglesize] }, i);
        }
        assertEquals(store.getNextSequenceIndex(), 10 * shinglesize + 5);
        assertArrayEquals(temp, shingle, 1e-6);
        assertArrayEquals(store.get(9 * shinglesize + 5), shingle, 1e-6);
        // double[] answer = store.getInternalShingle();
        assertNotEquals(store.internalShingle, store.getInternalShingle());
        assertTrue(store.pointEquals(9 * shinglesize + 5, shingle));
        assertFalse(store.pointEquals(9 * shinglesize + 4, shingle));
        assertArrayEquals(store.getInternalShingle(), shingle, 1e-6);
        assertArrayEquals(store.transformIndices(new int[] { 0 }), new int[] { 5 });
        assertThrows(IllegalArgumentException.class, () -> store.transformIndices(new int[] { 1 }));
        assertThrows(IllegalArgumentException.class, () -> store.transformToShingledPoint(new double[] { 1, 2 }));
        assertArrayEquals(store.transformToShingledPoint(new double[] { 0.0 }),
                store.transformToShingledPoint(new double[] { -0.0 }), 1e-6);
    }

    @Test
    public void checkRotationAndCompact() {
        int shinglesize = 4;
        PointStoreDouble store = new PointStoreDouble.Builder().capacity(2 * shinglesize).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(shinglesize).internalShinglingEnabled(true)
                .internalRotationEnabled(true).currentStoreCapacity(1).build();
        for (int i = 0; i < 2 * shinglesize; i++) {
            store.add(new double[] { -i - 1 }, i);
        }
        for (int i = 0; i < 2 * shinglesize - shinglesize + 1; i++) {
            if (i != shinglesize - 1) {
                store.decrementRefCount(i);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> store.get(0));
        double[] test = new double[shinglesize];
        for (int i = 0; i < shinglesize; i++) {
            test[i] = -(i + shinglesize + 1);
        }
        test[shinglesize - 1] = -shinglesize;
        assertArrayEquals(store.get(shinglesize - 1), test, 1e-6);
        store.compact();
        for (int i = 2 * shinglesize; i < 4 * shinglesize - 1; i++) {
            store.add(new double[] { -i - 1 }, i);
        }
        assertThrows(IllegalStateException.class, () -> store.add(new double[] { -4 * shinglesize }, 0));
        for (int i = 0; i < 2 * shinglesize; i++) {
            if (i != shinglesize - 1) {
                store.decrementRefCount(i);
            }
        }
        assertEquals(store.toString(shinglesize - 1), Arrays.toString(test));
        for (int i = 4 * shinglesize; i < 6 * shinglesize - 1; i++) {
            store.add(new double[] { -i - 1 }, i);
        }
        assertThrows(IllegalStateException.class,
                () -> store.add(new double[] { -6 * shinglesize }, 6 * shinglesize - 1));
        store.decrementRefCount(shinglesize - 1);
        store.add(new double[] { -6 * shinglesize }, 6 * shinglesize - 1);
        store.compact();
    }

    @Test
    void CompactionTest() {
        int shinglesize = 2;
        PointStoreDouble store = new PointStoreDouble.Builder().capacity(6).dimensions(shinglesize)
                .shingleSize(shinglesize).indexCapacity(6).directLocationEnabled(false).internalShinglingEnabled(true)
                .build();

        store.add(new double[] { 0 }, 0L);
        for (int i = 0; i < 5; i++) {
            store.add(new double[] { i + 1 }, 0L);
        }
        int finalIndex = store.add(new double[] { 4 + 2 }, 0L);
        assertArrayEquals(store.get(finalIndex), new double[] { 5, 6 });
        store.decrementRefCount(1);
        store.decrementRefCount(2);
        int index = store.add(new double[] { 7 }, 0L);
        assertArrayEquals(store.get(index), new double[] { 6, 7 });
        store.decrementRefCount(index);
        assertTrue(store.size() < store.capacity);
        index = store.add(new double[] { 8 }, 0L);
        assertArrayEquals(store.get(index), new double[] { 7, 8 });
    }
}
