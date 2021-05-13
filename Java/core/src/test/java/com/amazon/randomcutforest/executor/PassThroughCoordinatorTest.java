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


package com.amazon.randomcutforest.executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.Collections;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PassThroughCoordinatorTest {
    private PassThroughCoordinator coordinator;

    @BeforeEach
    public void setUp() {
        coordinator = new PassThroughCoordinator();
    }

    @Test
    public void testInitUpdate() {
        double[] point = new double[] {1.1, -2.2, 30.30};
        assertSame(point, coordinator.initUpdate(point, 0));
    }

    @Test
    public void testCompleteUpdate() {
        int totalUpdates = 10;
        for (int i = 0; i < totalUpdates; i++) {
            coordinator.completeUpdate(Collections.emptyList(), new double[] {i});
        }
        assertEquals(totalUpdates, coordinator.getTotalUpdates());
    }
}
