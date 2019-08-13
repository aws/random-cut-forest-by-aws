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

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CutTest {

    private int splitDimension;
    private double splitValue;
    private Cut cut;

    @BeforeEach
    public void setUp() {
        splitDimension = 2;
        splitValue = 3.4;
        cut = new Cut(splitDimension, splitValue);
    }

    @Test
    public void testNew() {
        assertThat(cut.getDimension(), is(splitDimension));
        assertThat(cut.getValue(), is(splitValue));
    }

    @Test
    public void testIsLeftOf() {
        double[] point = new double[] {1.0, 2.0, 3.0, 4.0};
        assertTrue(Cut.isLeftOf(point, cut));

        point[2] = 99.9;
        assertFalse(Cut.isLeftOf(point, cut));
    }
}
