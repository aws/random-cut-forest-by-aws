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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class BoundingBoxTest {

    private float[] point1;
    private float[] point2;
    private BoundingBox box1;
    private BoundingBox box2;

    @BeforeEach
    public void setUp() {
        point1 = new float[] { 1.5f, 2.7f };
        point2 = new float[] { 3.0f, 1.2f };
        box1 = new BoundingBox(point1);
        box2 = new BoundingBox(point2);
    }

    @Test
    public void testNewFromSinglePoint() {
        assertThat(box1.getDimensions(), is(2));
        assertThat((float) box1.getMinValue(0), is(point1[0]));
        assertThat((float) box1.getMaxValue(0), is(point1[0]));
        assertThat(box1.getRange(0), is(0.0));
        assertThat((float) box1.getMinValue(1), is(point1[1]));
        assertThat((float) box1.getMaxValue(1), is(point1[1]));
        assertThat(box1.getRange(1), is(0.0));
        assertThat(box1.getRangeSum(), is(0.0));

        assertThat(box2.getDimensions(), is(2));
        assertThat((float) box2.getMinValue(0), is(point2[0]));
        assertThat((float) box2.getMaxValue(0), is(point2[0]));
        assertThat(box2.getRange(0), is(0.0));
        assertThat((float) box2.getMinValue(1), is(point2[1]));
        assertThat((float) box2.getMaxValue(1), is(point2[1]));
        assertThat(box2.getRange(1), is(0.0));
        assertThat(box2.getRangeSum(), is(0.0));
    }

    @Test
    public void testGetMergedBoxWithOtherBox() {
        BoundingBox mergedBox = box1.getMergedBox(box2);

        assertThat(mergedBox.getDimensions(), is(2));
        assertThat((float) mergedBox.getMinValue(0), is(1.5f));
        assertThat((float) mergedBox.getMaxValue(0), is(3.0f));
        assertThat(mergedBox.getRange(0), closeTo(3.0 - 1.5, EPSILON));
        assertThat((float) mergedBox.getMinValue(1), is(1.2f));
        assertThat((float) mergedBox.getMaxValue(1), is(2.7f));
        assertThat(mergedBox.getRange(1), closeTo(2.7 - 1.2, EPSILON));

        double rangeSum = (3.0 - 1.5) + (2.7 - 1.2);
        assertThat(mergedBox.getRangeSum(), closeTo(rangeSum, EPSILON));

        // check that box1 and box2 were not changed

        assertThat(box1.getDimensions(), is(2));
        assertThat((float) box1.getMinValue(0), is(point1[0]));
        assertThat((float) box1.getMaxValue(0), is(point1[0]));
        assertThat(box1.getRange(0), is(0.0));
        assertThat((float) box1.getMinValue(1), is(point1[1]));
        assertThat((float) box1.getMaxValue(1), is(point1[1]));
        assertThat(box1.getRange(1), is(0.0));
        assertThat(box1.getRangeSum(), is(0.0));

        assertThat(box2.getDimensions(), is(2));
        assertThat((float) box2.getMinValue(0), is(point2[0]));
        assertThat((float) box2.getMaxValue(0), is(point2[0]));
        assertThat(box2.getRange(0), is(0.0));
        assertThat((float) box2.getMinValue(1), is(point2[1]));
        assertThat((float) box2.getMaxValue(1), is(point2[1]));
        assertThat(box2.getRange(1), is(0.0));
        assertThat(box2.getRangeSum(), is(0.0));
    }

    @Test
    public void testContainsBoundingBox() {
        BoundingBox box1 = new BoundingBox(new float[] { 0.0f, 0.0f })
                .getMergedBox(new BoundingBox(new float[] { 10.0f, 10.0f }));

        BoundingBox box2 = new BoundingBox(new float[] { 2.0f, 2.0f })
                .getMergedBox(new BoundingBox(new float[] { 8.0f, 8.0f }));

        BoundingBox box3 = new BoundingBox(new float[] { -4.0f, -4.0f })
                .getMergedBox(new BoundingBox(new float[] { -1.0f, -1.0f }));

        BoundingBox box4 = new BoundingBox(new float[] { 1.0f, -1.0f })
                .getMergedBox(new BoundingBox(new float[] { 5.0f, 5.0f }));

        // completely contains
        assertTrue(box1.contains(box2));
        assertFalse(box2.contains(box1));

        // completely disjoint
        assertFalse(box1.contains(box3));
        assertFalse(box3.contains(box1));

        // partially intersect
        assertFalse(box1.contains(box4));
        assertFalse(box4.contains(box1));
    }

    @Test
    public void testContainsPoint() {
        BoundingBox box1 = new BoundingBox(new float[] { 0.0f, 0.0f })
                .getMergedBox(new BoundingBox(new float[] { 10.0f, 10.0f }));

        assertTrue(box1.contains(new float[] { 0.0f, 0.1f }));
        assertTrue(box1.contains(new float[] { 5.5f, 6.5f }));
        assertFalse(box1.contains(new float[] { -0.7f, -4.5f }));
        assertFalse(box1.contains(new float[] { 5.0f, 11.0f }));
    }

}
