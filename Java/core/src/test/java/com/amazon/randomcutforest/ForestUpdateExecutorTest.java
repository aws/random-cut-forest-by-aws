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

package com.amazon.randomcutforest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.util.ArrayList;
import java.util.Arrays;
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

@ExtendWith(MockitoExtension.class)
public class ForestUpdateExecutorTest {

    private static final int numberOfTrees = 10;
    private static final int threadPoolSize = 2;

    @Captor
    private ArgumentCaptor<Sequential<double[]>> captor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ArrayList<IUpdatable<Sequential<double[]>>> sequentialTrees = new ArrayList<>();
            ArrayList<IUpdatable<Sequential<double[]>>> parallelTrees = new ArrayList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialTrees.add(mock(IUpdatable.class));
                parallelTrees.add(mock(IUpdatable.class));
            }

            IUpdateCoordinator<Sequential<double[]>> sequentialUpdateCoordinator = new PointSequencer();
            AbstractForestUpdateExecutor<Sequential<double[]>> sequentialExecutor = new SequentialForestUpdateExecutor<>(
                    sequentialUpdateCoordinator, sequentialTrees);

            IUpdateCoordinator<Sequential<double[]>> parallelUpdateCoordinator = new PointSequencer();
            AbstractForestUpdateExecutor<Sequential<double[]>> parallelExecutor = new ParallelForestUpdateExecutor<>(
                    parallelUpdateCoordinator, parallelTrees, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdate(AbstractForestUpdateExecutor<Sequential<double[]>> executor) {
        int totalUpdates = 10;
        List<double[]> expectedPoints = new ArrayList<>();

        for (int i = 0; i < totalUpdates; i++) {
            double[] point = new double[] { Math.sin(i), Math.cos(i) };
            executor.update(point);
            expectedPoints.add(point);
        }

        for (IUpdatable<Sequential<double[]>> tree : executor.models) {
            verify(tree, times(totalUpdates)).update(captor.capture());
            List<Sequential<double[]>> actualArguments = new ArrayList<>(captor.getAllValues());
            for (int i = 0; i < totalUpdates; i++) {
                Sequential<double[]> actual = actualArguments.get(i);
                assertEquals(i + 1, actual.getSequenceIndex());
                assertTrue(Arrays.equals(expectedPoints.get(i), actual.getValue()));
            }
        }

        assertEquals(totalUpdates, executor.getTotalUpdates());
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdateWithSignedZero(AbstractForestUpdateExecutor<Sequential<double[]>> executor) {
        double[] negativeZero = new double[] { -0.0, 0.0, 5.0 };
        double[] positiveZero = new double[] { 0.0, 0.0, 5.0 };

        executor.update(negativeZero);
        executor.update(positiveZero);

        for (IUpdatable<Sequential<double[]>> tree : executor.models) {
            verify(tree, times(2)).update(captor.capture());
            List<Sequential<double[]>> arguments = captor.getAllValues();

            Sequential<double[]> actual = arguments.get(0);
            assertEquals(1, actual.getSequenceIndex());
            assertTrue(Arrays.equals(positiveZero, actual.getValue()));

            actual = arguments.get(1);
            assertEquals(2, actual.getSequenceIndex());
            assertTrue(Arrays.equals(positiveZero, actual.getValue()));
        }
    }
}
