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
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import com.amazon.randomcutforest.store.PointStore;

public class PointStoreCoordinatorTest {

    private PointStore store;
    private PointStoreCoordinator coordinator;

    @BeforeEach
    public void setUp() {
        store = mock(PointStore.class);
        coordinator = new PointStoreCoordinator(store);
    }

    @Test
    public void testInitUpdate() {
        float[] point = { 1.2f, -3.4f };
        int index = 123;

        ArgumentCaptor<float[]> captor = ArgumentCaptor.forClass(float[].class);
        when(store.add(captor.capture(), anyLong())).thenReturn(index);

        int result = coordinator.initUpdate(point, 0);

        verify(store, times(1)).add(point, 0);
        assertEquals(result, index);
    }

    @Test
    public void testCompleteUpdate() {
        List<UpdateResult<Integer>> updateResults = new ArrayList<>();

        UpdateResult<Integer> result1 = UpdateResult.<Integer>builder().addedPoint(1).deletedPoint(100).build();
        updateResults.add(result1);

        UpdateResult<Integer> result2 = UpdateResult.<Integer>builder().addedPoint(2).deletedPoint(200).build();
        updateResults.add(result2);

        UpdateResult<Integer> result3 = UpdateResult.<Integer>builder().addedPoint(3).build();
        updateResults.add(result3);

        UpdateResult<Integer> result4 = UpdateResult.noop();
        updateResults.add(result4);

        // order shouldn't matter
        Collections.shuffle(updateResults);

        Integer updateInput = 1000;
        coordinator.completeUpdate(updateResults, updateInput);

        ArgumentCaptor<Integer> captor1 = ArgumentCaptor.forClass(Integer.class);
        verify(store, times(3)).incrementRefCount(captor1.capture());
        List<Integer> arguments = captor1.getAllValues();
        Collections.sort(arguments);
        assertEquals(1, arguments.get(0));
        assertEquals(2, arguments.get(1));
        assertEquals(3, arguments.get(2));

        ArgumentCaptor<Integer> captor2 = ArgumentCaptor.forClass(Integer.class);
        verify(store, times(3)).decrementRefCount(captor2.capture());
        arguments = captor2.getAllValues();
        Collections.sort(arguments);
        assertEquals(100, arguments.get(0));
        assertEquals(200, arguments.get(1));
        assertEquals(1000, arguments.get(2));
    }
}
