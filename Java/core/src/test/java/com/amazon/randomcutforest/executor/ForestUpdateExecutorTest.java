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

import static org.mockito.Mockito.mock;

import java.util.stream.Stream;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
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
    private ArgumentCaptor<double[]> captor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ComponentList<double[]> sequentialComponents = new ComponentList<>();
            ComponentList<double[]> parallelComponents = new ComponentList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialComponents.add(mock(IComponentModel.class));
                parallelComponents.add(mock(IComponentModel.class));
            }

            IUpdateCoordinator<double[]> sequentialUpdateCoordinator = new PassThroughCoordinator();
            AbstractForestUpdateExecutor<double[]> sequentialExecutor = new SequentialForestUpdateExecutor<>(
                    sequentialUpdateCoordinator, sequentialComponents);

            IUpdateCoordinator<double[]> parallelUpdateCoordinator = new PassThroughCoordinator();
            AbstractForestUpdateExecutor<double[]> parallelExecutor = new ParallelForestUpdateExecutor<>(
                    parallelUpdateCoordinator, parallelComponents, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }
}
