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

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atMost;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.amazon.randomcutforest.ComponentList;
import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.TestUtils;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

public class ForestTraversalExecutorTest {

    private static int numberOfTrees = 10;
    private static int threadPoolSize = 2;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context)
                throws Exception {

            ComponentList<double[], double[]> sequentialExecutors = new ComponentList<>();
            ComponentList<double[], double[]> parallelExecutors = new ComponentList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                SimpleStreamSampler<double[]> sampler = mock(SimpleStreamSampler.class);
                RandomCutTree tree = mock(RandomCutTree.class);
                sequentialExecutors.add(spy(new SamplerPlusTree<>(sampler, tree)));
            }

            for (int i = 0; i < numberOfTrees; i++) {
                SimpleStreamSampler<double[]> sampler = mock(SimpleStreamSampler.class);
                RandomCutTree tree = mock(RandomCutTree.class);
                parallelExecutors.add(spy(new SamplerPlusTree<>(sampler, tree)));
            }

            SequentialForestTraversalExecutor sequentialExecutor =
                    new SequentialForestTraversalExecutor(sequentialExecutors);

            ParallelForestTraversalExecutor parallelExecutor =
                    new ParallelForestTraversalExecutor(parallelExecutors, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestBinaryAccumulator(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) executor.components.get(i)).getTree();
            when(tree.traverse(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result =
                executor.traverseForest(
                        point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, Double::sum, x -> x / 10.0);

        for (IComponentModel<?, ?> component : executor.components) {
            verify(component, times(1)).traverse(aryEq(point), any());
        }

        assertEquals(expectedResult, result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestCollector(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double[] expectedResult = new double[numberOfTrees];

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) executor.components.get(i)).getTree();
            when(tree.traverse(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result =
                executor.traverseForest(
                        point,
                        TestUtils.DUMMY_GENERIC_VISITOR_FACTORY,
                        TestUtils.SORTED_LIST_COLLECTOR);

        for (IComponentModel<?, ?> component : executor.components) {
            verify(component, times(1)).traverse(aryEq(point), any());
        }

        assertEquals(numberOfTrees, result.size());
        for (int i = 0; i < numberOfTrees; i++) {
            assertEquals(expectedResult[i], result.get(i), EPSILON);
        }
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestConverging(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) executor.components.get(i)).getTree();
            when(tree.traverse(aryEq(point), any())).thenReturn(treeResult);
        }

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        double result =
                executor.traverseForest(
                        point,
                        TestUtils.DUMMY_GENERIC_VISITOR_FACTORY,
                        accumulator,
                        x -> x / accumulator.getValuesAccepted());

        for (IComponentModel<?, ?> component : executor.components) {
            verify(component, atMost(1)).traverse(aryEq(point), any());
        }

        assertTrue(accumulator.getValuesAccepted() >= convergenceThreshold);
        assertTrue(accumulator.getValuesAccepted() < numberOfTrees);
        assertEquals(
                accumulator.getAccumulatedValue() / accumulator.getValuesAccepted(),
                result,
                EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestMultiBinaryAccumulator(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) executor.components.get(i)).getTree();
            when(tree.traverseMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result =
                executor.traverseForestMulti(
                        point,
                        TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                        Double::sum,
                        x -> x / 10.0);

        for (IComponentModel<?, ?> component : executor.components) {
            verify(component, times(1)).traverseMulti(aryEq(point), any());
        }

        assertEquals(expectedResult, result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestMultiCollector(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double[] expectedResult = new double[numberOfTrees];

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) executor.components.get(i)).getTree();
            when(tree.traverseMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result =
                executor.traverseForestMulti(
                        point,
                        TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                        TestUtils.SORTED_LIST_COLLECTOR);

        for (IComponentModel<?, ?> component : executor.components) {
            verify(component, times(1)).traverseMulti(aryEq(point), any());
        }

        assertEquals(numberOfTrees, result.size());
        for (int i = 0; i < numberOfTrees; i++) {
            assertEquals(expectedResult[i], result.get(i), EPSILON);
        }
    }
}
