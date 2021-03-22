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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.mockito.AdditionalMatchers.aryEq;
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

@ExtendWith(MockitoExtension.class)
public class ForestUpdateExecutorTest {

    private static final int numberOfTrees = 10;
    private static final int threadPoolSize = 2;

    @Captor
    private ArgumentCaptor<List<UpdateResult<double[]>>> updateResultCaptor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ComponentList<double[], double[]> sequentialComponents = new ComponentList<>();
            ComponentList<double[], double[]> parallelComponents = new ComponentList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialComponents.add(mock(IComponentModel.class));
                parallelComponents.add(mock(IComponentModel.class));
            }

            IUpdateCoordinator<double[], double[]> sequentialUpdateCoordinator = spy(new PassThroughCoordinator());
            AbstractForestUpdateExecutor<double[], double[]> sequentialExecutor = new SequentialForestUpdateExecutor<>(
                    sequentialUpdateCoordinator, sequentialComponents);

            IUpdateCoordinator<double[], double[]> parallelUpdateCoordinator = spy(new PassThroughCoordinator());
            AbstractForestUpdateExecutor<double[], double[]> parallelExecutor = new ParallelForestUpdateExecutor<>(
                    parallelUpdateCoordinator, parallelComponents, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdate(AbstractForestUpdateExecutor<double[], ?> executor) {
        int addAndDelete = 4;
        int addOnly = 4;

        ComponentList<double[], ?> components = executor.components;
        for (int i = 0; i < addAndDelete; i++) {
            IComponentModel<double[], ?> model = components.get(i);
            UpdateResult<double[]> result = new UpdateResult<>(new double[] { i }, new double[] { 2 * i });
            when(model.update(any(), anyLong())).thenReturn(result);
        }

        for (int i = addAndDelete; i < addAndDelete + addOnly; i++) {
            IComponentModel<double[], ?> model = components.get(i);
            UpdateResult<double[]> result = UpdateResult.<double[]>builder().addedPoint(new double[] { i }).build();
            when(model.update(any(), anyLong())).thenReturn(result);
        }

        for (int i = addAndDelete + addOnly; i < numberOfTrees; i++) {
            IComponentModel<double[], ?> model = components.get(i);
            when(model.update(any(), anyLong())).thenReturn(UpdateResult.noop());
        }

        double[] point = new double[] { 1.0 };
        executor.update(point);

        executor.components.forEach(model -> verify(model).update(aryEq(point), eq(0L)));

        IUpdateCoordinator<double[], ?> coordinator = executor.updateCoordinator;
        verify(coordinator, times(1)).completeUpdate(updateResultCaptor.capture(), aryEq(point));

        List<UpdateResult<double[]>> updateResults = updateResultCaptor.getValue();
        assertEquals(addAndDelete + addOnly, updateResults.size());

        int actualAddAndAndDelete = 0;
        int actualAddOnly = 0;
        for (int i = 0; i < updateResults.size(); i++) {
            UpdateResult<double[]> result = updateResults.get(i);
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
        double[] point1 = new double[] { 1.0, -22.2, 30.9 };
        double[] point1Copy = executor.cleanCopy(point1);
        assertNotSame(point1, point1Copy);
        assertArrayEquals(point1, point1Copy);

        double[] point2 = new double[] { -0.0, -22.2, 30.9 };
        double[] point2Copy = executor.cleanCopy(point2);
        assertNotSame(point2, point2Copy);
        assertEquals(0.0, point2Copy[0]);

        point2Copy[0] = -0.0;
        assertArrayEquals(point2, point2Copy);
    }
}
