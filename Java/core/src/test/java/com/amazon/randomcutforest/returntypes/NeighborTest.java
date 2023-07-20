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

package com.amazon.randomcutforest.returntypes;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

public class NeighborTest {
    @Test
    public void testNew() {
        float[] point = new float[] { 1.0f, -2.0f, 3.3f };
        double distance = 1234.5;
        List<Long> timestamps = new ArrayList<>();
        timestamps.add(99999L);
        timestamps.add(99L);
        Neighbor neighbor = new Neighbor(point, distance, timestamps);

        assertArrayEquals(point, neighbor.point);
        assertEquals(distance, neighbor.distance);
        assertThat(neighbor.sequenceIndexes, containsInAnyOrder(timestamps.toArray()));
    }
}
