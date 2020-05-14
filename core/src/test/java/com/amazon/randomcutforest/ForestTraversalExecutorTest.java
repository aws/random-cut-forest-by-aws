package com.amazon.randomcutforest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.RandomCutTree;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;
import org.mockito.Mockito;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.atMost;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ForestTraversalExecutorTest {

    private static int numberOfTrees = 10;
    private static int threadPoolSize = 2;

    private static class TestExecutorProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext context) throws Exception {

            ArrayList<TreeUpdater> sequentialTreeUpdaters = new ArrayList<>();
            ArrayList<TreeUpdater> parallelTreeUpdaters = new ArrayList<>();

            for (int i = 0; i < numberOfTrees; i++) {
                SimpleStreamSampler sampler = mock(SimpleStreamSampler.class);
                RandomCutTree tree = mock(RandomCutTree.class);
                sequentialTreeUpdaters.add(Mockito.spy(new TreeUpdater(sampler, tree)));
            }

            for (int i = 0; i < numberOfTrees; i++) {
                SimpleStreamSampler sampler = mock(SimpleStreamSampler.class);
                RandomCutTree tree = mock(RandomCutTree.class);
                parallelTreeUpdaters.add(Mockito.spy(new TreeUpdater(sampler, tree)));
            }

            SequentialForestTraversalExecutor sequentialExecutor =
                    new SequentialForestTraversalExecutor(sequentialTreeUpdaters);

            ParallelForestTraversalExecutor parallelExecutor =
                    new ParallelForestTraversalExecutor(parallelTreeUpdaters, threadPoolSize);

            return Stream.of(sequentialExecutor, parallelExecutor).map(Arguments::of);
        }
    }


    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testUpdate(AbstractForestTraversalExecutor executor) {
        int totalUpdates = 10;
        for (int i = 0; i < totalUpdates; i++) {
            double[] point = new double[] {Math.sin(i), Math.cos(i)};
            executor.update(point);

            for (TreeUpdater updater: executor.treeUpdaters) {
                verify(updater, times(1)).update(point, i + 1);
            }
        }

        assertEquals(totalUpdates, executor.getTotalUpdates());
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestBinaryAccumulator(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            RandomCutTree tree = executor.treeUpdaters.get(i).getTree();
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result = executor.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, Double::sum, x -> x / 10.0);

        for (TreeUpdater updater: executor.treeUpdaters) {
            verify(updater.getTree(), times(1)).traverseTree(aryEq(point), any());
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
            RandomCutTree tree = executor.treeUpdaters.get(i).getTree();
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result = executor.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);

        for (TreeUpdater updater: executor.treeUpdaters) {
            verify(updater.getTree(), times(1)).traverseTree(aryEq(point), any());
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
            RandomCutTree tree = executor.treeUpdaters.get(i).getTree();
            when(tree.traverseTree(aryEq(point), any())).thenReturn(treeResult);
        }

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        double result = executor.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator,
                x -> x / accumulator.getValuesAccepted());

        for (TreeUpdater updater: executor.treeUpdaters) {
            verify(updater.getTree(), atMost(1)).traverseTree(aryEq(point), any());
        }

        assertTrue(accumulator.getValuesAccepted() >= convergenceThreshold);
        assertTrue(accumulator.getValuesAccepted() < numberOfTrees);
        assertEquals(accumulator.getAccumulatedValue() / accumulator.getValuesAccepted(), result, EPSILON);
    }

    @ParameterizedTest
    @ArgumentsSource(TestExecutorProvider.class)
    public void testTraverseForestMultiBinaryAccumulator(AbstractForestTraversalExecutor executor) {
        double[] point = new double[] {1.2, -3.4};
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            double treeResult = Math.random();
            RandomCutTree tree = executor.treeUpdaters.get(i).getTree();
            when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;

        double result = executor.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, Double::sum,
                x -> x / 10.0);

        for (TreeUpdater updater: executor.treeUpdaters) {
            verify(updater.getTree(), times(1)).traverseTreeMulti(aryEq(point), any());
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
            RandomCutTree tree = executor.treeUpdaters.get(i).getTree();
            when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(treeResult);
            expectedResult[i] = treeResult;
        }

        Arrays.sort(expectedResult);

        List<Double> result = executor.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);

        for (TreeUpdater updater: executor.treeUpdaters) {
            verify(updater.getTree(), times(1)).traverseTreeMulti(aryEq(point), any());
        }

        assertEquals(numberOfTrees, result.size());
        for (int i = 0; i < numberOfTrees; i++) {
            assertEquals(expectedResult[i], result.get(i), EPSILON);
        }
    }
}
