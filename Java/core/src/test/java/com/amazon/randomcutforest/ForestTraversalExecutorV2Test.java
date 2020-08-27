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

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atMost;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.IUpdatableTree;

@ExtendWith(MockitoExtension.class)
public class ForestTraversalExecutorV2Test {

    private static final int numberOfTrees = 10;
    private static final int threadPoolSize = 2;

    @Captor
    private ArgumentCaptor<Sequential<double[]>> captor;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ArrayList<IUpdatableTree<Sequential<double[]>>> sequentialTrees = new ArrayList<>();
            ArrayList<IUpdatableTree<Sequential<double[]>>> parallelTrees = new ArrayList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                sequentialTrees.add(mock(IUpdatableTree.class));
                parallelTrees.add(mock(IUpdatableTree.class));
            }

            IUpdateCoordinator<Sequential<double[]>> sequentialUpdateCoordinator = new PointSequencer();
            GenericForestTraversalExecutor<Sequential<double[]>> sequentialExecutor = new SequentialForestTraversalExecutorV2<>(
                    sequentialUpdateCoordinator, sequentialTrees);

            IUpdateCoordinator<Sequential<double[]>> parallelUpdateCoordinator = new PointSequencer();
            GenericForestTraversalExecutor<Sequential<double[]>> parallelExecutor = new ParallelForestTraversalExecutorV2<>(
                    parallelUpdateCoordinator, parallelTrees, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdate(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        int totalUpdates = 10;
        List<double[]> expectedPoints = new ArrayList<>();

        for (int i = 0; i < totalUpdates; i++) {
            double[] point = new double[] { Math.sin(i), Math.cos(i) };
            executor.update(point);
            expectedPoints.add(point);
        }

        for (IUpdatableTree<Sequential<double[]>> tree : executor.trees) {
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
    public void testUpdateWithSignedZero(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] negativeZero = new double[] { -0.0, 0.0, 5.0 };
        double[] positiveZero = new double[] { 0.0, 0.0, 5.0 };

        executor.update(negativeZero);
        executor.update(positiveZero);

        for (IUpdatableTree<Sequential<double[]>> tree : executor.trees) {
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

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestBinaryAccumulator(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] point = new double[] { 1.2, -3.4 };
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?> tree = executor.trees.get(i);
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result = executor.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, Double::sum,
                x -> x / 10.0);

        for (ITree<?> tree : executor.trees) {
            verify(tree, times(1)).traverseTree(aryEq(point), any());
        }

        assertEquals(expectedResult, result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestCollector(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] point = new double[] { 1.2, -3.4 };
        double[] expectedResult = new double[numberOfTrees];

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?> tree = executor.trees.get(i);
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result = executor.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);

        for (ITree<?> tree : executor.trees) {
            verify(tree, times(1)).traverseTree(aryEq(point), any());
        }

        assertEquals(numberOfTrees, result.size());
        for (int i = 0; i < numberOfTrees; i++) {
            assertEquals(expectedResult[i], result.get(i), EPSILON);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestConverging(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] point = new double[] { 1.2, -3.4 };

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?> tree = executor.trees.get(i);
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
        }

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        double result = executor.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator,
                x -> x / accumulator.getValuesAccepted());

        for (ITree<?> tree : executor.trees) {
            verify(tree, atMost(1)).traverseTree(aryEq(point), any());
        }

        assertTrue(accumulator.getValuesAccepted() >= convergenceThreshold);
        assertTrue(accumulator.getValuesAccepted() < numberOfTrees);
        assertEquals(accumulator.getAccumulatedValue() / accumulator.getValuesAccepted(), result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestMultiBinaryAccumulator(
            GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] point = new double[] { 1.2, -3.4 };
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?> tree = executor.trees.get(i);
            when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result = executor.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, Double::sum,
                x -> x / 10.0);

        for (ITree<?> tree : executor.trees) {
            verify(tree, times(1)).traverseTreeMulti(aryEq(point), any());
        }

        assertEquals(expectedResult, result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestMultiCollector(GenericForestTraversalExecutor<Sequential<double[]>> executor) {
        double[] point = new double[] { 1.2, -3.4 };
        double[] expectedResult = new double[numberOfTrees];

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?> tree = executor.trees.get(i);
            when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result = executor.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);

        for (ITree<?> tree : executor.trees) {
            verify(tree, times(1)).traverseTreeMulti(aryEq(point), any());
        }

        assertEquals(numberOfTrees, result.size());
        for (int i = 0; i < numberOfTrees; i++) {
            assertEquals(expectedResult[i], result.get(i), EPSILON);
        }
    }
}
