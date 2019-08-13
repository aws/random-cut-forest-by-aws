/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BoundingBoxTest {

    private double[] point1;
    private double[] point2;
    private BoundingBox box1;
    private BoundingBox box2;

    @BeforeEach
    public void setUp() {
        point1 = new double[] {1.5, 2.7};
        point2 = new double[] {3.0, 1.2};
        box1 = new BoundingBox(point1);
        box2 = new BoundingBox(point2);
    }


    @Test
    public void testNewFromSinglePoint() {
        assertThat(box1.getDimensions(), is(2));
        assertThat(box1.getMinValue(0), is(point1[0]));
        assertThat(box1.getMaxValue(0), is(point1[0]));
        assertThat(box1.getRange(0), is(0.0));
        assertThat(box1.getMinValue(1), is(point1[1]));
        assertThat(box1.getMaxValue(1), is(point1[1]));
        assertThat(box1.getRange(1), is(0.0));
        assertThat(box1.getRangeSum(), is(0.0));

        assertThat(box2.getDimensions(), is(2));
        assertThat(box2.getMinValue(0), is(point2[0]));
        assertThat(box2.getMaxValue(0), is(point2[0]));
        assertThat(box2.getRange(0), is(0.0));
        assertThat(box2.getMinValue(1), is(point2[1]));
        assertThat(box2.getMaxValue(1), is(point2[1]));
        assertThat(box2.getRange(1), is(0.0));
        assertThat(box2.getRangeSum(), is(0.0));
    }

    @Test
    public void testGetMergedBoxWithOtherBox() {
        BoundingBox mergedBox = box1.getMergedBox(box2);

        assertThat(mergedBox.getDimensions(), is(2));
        assertThat(mergedBox.getMinValue(0), is(1.5));
        assertThat(mergedBox.getMaxValue(0), is(3.0));
        assertThat(mergedBox.getRange(0), closeTo(3.0 - 1.5, EPSILON));
        assertThat(mergedBox.getMinValue(1), is(1.2));
        assertThat(mergedBox.getMaxValue(1), is(2.7));
        assertThat(mergedBox.getRange(1), closeTo(2.7 - 1.2, EPSILON));

        double rangeSum = (3.0 - 1.5) + (2.7 - 1.2);
        assertThat(mergedBox.getRangeSum(), closeTo(rangeSum, EPSILON));

        // check that box1 and box2 were not changed

        assertThat(box1.getDimensions(), is(2));
        assertThat(box1.getMinValue(0), is(point1[0]));
        assertThat(box1.getMaxValue(0), is(point1[0]));
        assertThat(box1.getRange(0), is(0.0));
        assertThat(box1.getMinValue(1), is(point1[1]));
        assertThat(box1.getMaxValue(1), is(point1[1]));
        assertThat(box1.getRange(1), is(0.0));
        assertThat(box1.getRangeSum(), is(0.0));

        assertThat(box2.getDimensions(), is(2));
        assertThat(box2.getMinValue(0), is(point2[0]));
        assertThat(box2.getMaxValue(0), is(point2[0]));
        assertThat(box2.getRange(0), is(0.0));
        assertThat(box2.getMinValue(1), is(point2[1]));
        assertThat(box2.getMaxValue(1), is(point2[1]));
        assertThat(box2.getRange(1), is(0.0));
        assertThat(box2.getRangeSum(), is(0.0));
    }

    @Test
    public void testContainsBoundingBox() {
        BoundingBox box1 = new BoundingBox(new double[] {0.0, 0.0})
                .getMergedBox(new BoundingBox(new double[] {10.0, 10.0}));

        BoundingBox box2 = new BoundingBox(new double[] {2.0, 2.0})
                .getMergedBox(new BoundingBox(new double[] {8.0, 8.0}));

        BoundingBox box3 = new BoundingBox(new double[] {-4.0, -4.0})
                .getMergedBox(new BoundingBox(new double[] {-1.0, -1.0}));

        BoundingBox box4 = new BoundingBox(new double[] {1.0, -1.0})
                .getMergedBox(new BoundingBox(new double[] {5.0, 5.0}));

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
        BoundingBox box1 = new BoundingBox(new double[] {0.0, 0.0})
                .getMergedBox(new BoundingBox(new double[] {10.0, 10.0}));

        assertTrue(box1.contains(new double[] {0.0, 0.1}));
        assertTrue(box1.contains(new double[] {5.5, 6.5}));
        assertFalse(box1.contains(new double[] {-0.7, -4.5}));
        assertFalse(box1.contains(new double[] {5.0, 11.0}));
    }

    @Test
    public void testEqualsAndHashCode() {
        double[] minValues1 = new double[] { -0.1, 0.1};
        double[] maxValues1 = new double[] { -0.1, 0.1};

        double[] minValues2 = new double[] { -0.1 + 1e-8, 0.1};
        double[] maxValues2 = new double[] { -0.1, 0.1 - 1e-8};

        double[] minValues3 = new double[] { -0.1, 0.1, 100};
        double[] maxValues3 = new double[] { -0.1, 0.1, 4};

        BoundingBox box1 = new BoundingBox(minValues1).getMergedBox(maxValues1);
        BoundingBox box2 = new BoundingBox(minValues1).getMergedBox(maxValues1);
        BoundingBox box3 = new BoundingBox(minValues2).getMergedBox(maxValues1);
        BoundingBox box4 = new BoundingBox(minValues1).getMergedBox(maxValues2);
        BoundingBox box5 = new BoundingBox(minValues3).getMergedBox(maxValues3);

        assertEquals(box1, box2);
        assertEquals(box1.hashCode(), box2.hashCode());

        assertNotEquals(box1, box3);
        assertNotEquals(box1.hashCode(), box3.hashCode());

        assertNotEquals(box1, box4);
        assertNotEquals(box1.hashCode(), box4.hashCode());

        assertNotEquals(box1, box5);
        assertNotEquals(box1.hashCode(), box5.hashCode());

        assertNotEquals(box1, new Object());
    }
}
