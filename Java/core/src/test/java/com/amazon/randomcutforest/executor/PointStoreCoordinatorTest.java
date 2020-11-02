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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import com.amazon.randomcutforest.store.PointStoreDouble;

public class PointStoreCoordinatorTest {

    private PointStoreDouble store;
    private PointStoreCoordinator coordinator;

    @BeforeEach
    public void setUp() {
        store = mock(PointStoreDouble.class);
        coordinator = new PointStoreCoordinator(store);
    }

    @Test
    public void testInitUpdate() {
        double[] point = { 1.2, -3.4 };
        int index = 123;

        ArgumentCaptor<double[]> captor = ArgumentCaptor.forClass(double[].class);
        when(store.add(captor.capture())).thenReturn(index);

        int result = coordinator.initUpdate(point);

        verify(store, times(1)).add(point);
        assertEquals(result, index);
    }

    @Test
    public void testCompleteUpdate() {

    }
}
