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

import static org.mockito.Mockito.*;

import java.util.ArrayList;
import java.util.stream.Stream;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ForestUpdateExecutorTest {

    private static final int numberOfTrees = 10;
    private static final int threadPoolSize = 2;

    @Captor
    private ArgumentCaptor<double[]> captor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ArrayList<IUpdatable<Sequential<double[]>>> sequentialTrees = new ArrayList<>();
            ArrayList<IUpdatable<Sequential<double[]>>> parallelTrees = new ArrayList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialTrees.add(mock(IUpdatable.class));
                parallelTrees.add(mock(IUpdatable.class));
            }

            IUpdateCoordinator<double[]> sequentialUpdateCoordinator = new PointSequencer();
            AbstractForestUpdateExecutor<Sequential<double[]>> sequentialExecutor = new SequentialForestUpdateExecutor(
                    sequentialUpdateCoordinator, sequentialTrees);

            IUpdateCoordinator<double[]> parallelUpdateCoordinator = new PointSequencer();
            AbstractForestUpdateExecutor<Sequential<double[]>> parallelExecutor = new ParallelForestUpdateExecutor(
                    parallelUpdateCoordinator, parallelTrees, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

}
