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

import static com.amazon.randomcutforest.util.ArrayUtils.cleanCopy;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.junit.jupiter.MockitoExtension;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.store.PointStore;

@ExtendWith(MockitoExtension.class)
public class ForestUpdateExecutorTest {

    private static final int numberOfTrees = 10;
    private static final int threadPoolSize = 2;

    @Captor
    private ArgumentCaptor<List<UpdateResult<Integer>>> updateResultCaptor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ComponentList<Integer, float[]> sequentialComponents = new ComponentList<>();
            ComponentList<Integer, float[]> parallelComponents = new ComponentList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialComponents.add(mock(IComponentModel.class));
                parallelComponents.add(mock(IComponentModel.class));
            }

            PointStore pointStore = mock(PointStore.class);
            IStateCoordinator<Integer, float[]> sequentialUpdateCoordinator = spy(
                    new PointStoreCoordinator<>(pointStore));
            AbstractForestUpdateExecutor<Integer, float[]> sequentialExecutor = new SequentialForestUpdateExecutor<>(
                    sequentialUpdateCoordinator, sequentialComponents);

            IStateCoordinator<Integer, float[]> parallelUpdateCoordinator = spy(
                    new PointStoreCoordinator<>(pointStore));
            AbstractForestUpdateExecutor<Integer, float[]> parallelExecutor = new ParallelForestUpdateExecutor<>(
                    parallelUpdateCoordinator, parallelComponents, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdate(AbstractForestUpdateExecutor<Integer, float[]> executor) {
        int addAndDelete = 4;
        int addOnly = 4;

        ComponentList<Integer, ?> components = executor.components;
        for (int i = 0; i < addAndDelete; i++) {
            IComponentModel<Integer, ?> model = components.get(i);
            UpdateResult<Integer> result = new UpdateResult<>(i, 2 * i);
            when(model.update(any(), anyLong())).thenReturn(result);
        }

        for (int i = addAndDelete; i < addAndDelete + addOnly; i++) {
            IComponentModel<Integer, ?> model = components.get(i);
            UpdateResult<Integer> result = UpdateResult.<Integer>builder().addedPoint(i).build();
            when(model.update(any(), anyLong())).thenReturn(result);
        }

        for (int i = addAndDelete + addOnly; i < numberOfTrees; i++) {
            IComponentModel<Integer, ?> model = components.get(i);
            when(model.update(any(), anyLong())).thenReturn(UpdateResult.noop());
        }

        float[] point = new float[] { 1.0f };
        executor.update(point);

        executor.components.forEach(model -> verify(model).update(any(), eq(0L)));

        IStateCoordinator<Integer, ?> coordinator = executor.updateCoordinator;
        verify(coordinator, times(1)).completeUpdate(updateResultCaptor.capture(), any());

        List<UpdateResult<Integer>> updateResults = updateResultCaptor.getValue();
        assertEquals(addAndDelete + addOnly, updateResults.size());

        int actualAddAndAndDelete = 0;
        int actualAddOnly = 0;
        for (int i = 0; i < updateResults.size(); i++) {
            UpdateResult<Integer> result = updateResults.get(i);
            if (result.getDeletedPoint().isPresent()) {
                actualAddAndAndDelete++;
            } else {
                actualAddOnly++;
            }
        }

        assertEquals(addAndDelete, actualAddAndAndDelete);
        assertEquals(addOnly, actualAddOnly);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testCleanCopy(AbstractForestUpdateExecutor<double[], ?> executor) {
        float[] point1 = new float[] { 1.0f, -22.2f, 30.9f };
        float[] point1Copy = cleanCopy(point1);
        assertNotSame(point1, point1Copy);
        assertArrayEquals(point1, point1Copy);

        float[] point2 = new float[] { -0.0f, -22.2f, 30.9f };
        float[] point2Copy = cleanCopy(point2);
        assertNotSame(point2, point2Copy);
        assertEquals(0.0, point2Copy[0]);

        point2Copy[0] = -0.0f;
        assertArrayEquals(point2, point2Copy);
    }
}
