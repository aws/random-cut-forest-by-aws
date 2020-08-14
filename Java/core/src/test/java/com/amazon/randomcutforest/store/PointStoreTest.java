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
        pointStore = new PointStore(dimensions, capacity);
    }

    @Test
    public void testNew() {
        assertEquals(dimensions, pointStore.getDimensions());
        assertEquals(capacity, pointStore.getCapacity());
        assertEquals(0, pointStore.size());

        for (int i = 0; i < capacity; i++) {
            assertEquals(0, pointStore.getRefCount(i));
        }
    }

    @Test
    public void testAdd() {
        float[] point1 = { 1.2f, -3.4f };
        int offset1 = pointStore.add(point1);
        assertTrue(offset1 >= 0 && offset1 < capacity);
        assertEquals(1, pointStore.getRefCount(offset1));
        assertEquals(1, pointStore.size());

        float[] retrievedPoint1 = pointStore.get(offset1);
        assertNotSame(point1, retrievedPoint1);
        assertArrayEquals(point1, retrievedPoint1);

        float[] point2 = { 111.2f, -333.4f };
        int offset2 = pointStore.add(point2);
        assertTrue(offset2 >= 0 && offset2 < capacity);
        assertEquals(1, pointStore.getRefCount(offset2));
        assertEquals(2, pointStore.size());
        assertNotEquals(offset1, offset2);

        float[] retrievedPoint2 = pointStore.get(offset2);
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
        assertThrows(IllegalArgumentException.class, () -> pointStore.add(new float[] { 1.1f, -2.2f, 3.3f }));

        for (int i = 0; i < capacity; i++) {
            float[] point = new float[dimensions];
            point[0] = (float) Math.random();
            point[1] = (float) Math.random();
            pointStore.add(point);
        }

        // point store is full
        assertThrows(IllegalStateException.class, () -> pointStore.add(new float[] { 1.1f, -2.2f }));
    }

    @Test
    public void testGetInvalid() {
        assertThrows(IllegalArgumentException.class, () -> pointStore.get(-1));
        assertThrows(IllegalArgumentException.class, () -> pointStore.get(capacity));
    }

    @Test
    public void testIncrementRefCount() {
        float[] point = { 1.2f, -3.4f };
        int offset = pointStore.add(point);
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
        int offset = pointStore.add(point);
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
        int offset = pointStore.add(point);
        assertTrue(pointStore.pointEquals(offset, point));
        assertFalse(pointStore.pointEquals(offset, new float[] { 5.6f, -7.8f }));
    }

    @Test
    public void testPointEqualsInvalid() {
        float[] point = { 1.2f, -3.4f };
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(-1, point));
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(0, point));

        int offset = pointStore.add(point);
        assertThrows(IllegalArgumentException.class, () -> pointStore.pointEquals(offset, new float[] { 99.9f }));
    }
}
