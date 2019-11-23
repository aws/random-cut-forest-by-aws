package com.amazon.randomcutforest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import com.amazon.randomcutforest.anomalydetection.AnomalyAttributionVisitor;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.imputation.ImputeVisitor;
import com.amazon.randomcutforest.interpolation.SimpleInterpolationVisitor;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDiVectorAccumulator;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDoubleAccumulator;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.Node;
import com.amazon.randomcutforest.tree.RandomCutTree;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.powermock.reflect.Whitebox;

import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class RandomCutForestTest {

    private int dimensions;
    private int sampleSize;
    private int numberOfTrees;
    private ArrayList<TreeUpdater> treeUpdaters;
    private AbstractForestTraversalExecutor executor;
    private RandomCutForest forest;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        sampleSize = 256;
        numberOfTrees = 10;

        treeUpdaters = new ArrayList<>();
        for (int i = 0; i < numberOfTrees; i++) {
            SimpleStreamSampler sampler = mock(SimpleStreamSampler.class);
            RandomCutTree tree = mock(RandomCutTree.class);
            treeUpdaters.add(spy(new TreeUpdater(sampler, tree)));
        }

        executor = spy(new SequentialForestTraversalExecutor(treeUpdaters));

        forest = RandomCutForest.builder()
                .dimensions(dimensions)
                .numberOfTrees(numberOfTrees)
                .sampleSize(sampleSize)
                .build();
        forest = spy(forest);
        Whitebox.setInternalState(forest, "executor", executor);
    }

    @Test
    public void testUpdate() {
        double[] point = {2.2, -1.1};
        forest.update(point);
        verify(executor, times(1)).update(point);
    }

    @Test
    public void testUpdateInvalid() {
        assertThrows(NullPointerException.class, () -> forest.update(null));
        assertThrows(IllegalArgumentException.class, () -> forest.update(new double[] {1.2, 3.4, -5.6}));
    }

    @Test
    public void testTraverseForestBinaryAccumulator() {
        double[] point = {2.2, -1.1};
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher);
        verify(executor, times(1)).traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY,
                accumulator, finisher);
    }

    @Test
    public void testTranverseForestBinaryAccumulatorInvalid() {
        double[] point = {2.2, -1.1};
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(null, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () ->
                forest.traverseForest(new double[] {2.2, -1.1, 3.3}, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, (BinaryOperator<Double>) null, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestCollector() {
        double[] point = {2.2, -1.1};

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR);
        verify(executor, times(1)).traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTranverseForestCollectorInvalid() {
        double[] point = {2.2, -1.1};

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(null, TestUtils.DUMMY_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () ->
                forest.traverseForest(new double[] {2.2, -1.1, 3.3}, TestUtils.DUMMY_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, null));
    }

    @Test
    public void testTraverseForestConverging() {
        double[] point = new double[] {1.2, -3.4};

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher);
        verify(executor, times(1)).traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher);
    }

    @Test
    public void testTraverseForestConvergingInvalid() {
        double[] point = new double[] {1.2, -3.4};

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(null, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () ->
                forest.traverseForest(new double[] {1.2, -3.4, 5.6}, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, (ConvergingAccumulator<Double>) null, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForest(point, TestUtils.DUMMY_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void traverseForestMultiBinaryAccumulator() {
        double[] point = {2.2, -1.1};
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(0.0);
                });

        forest.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, accumulator, finisher);
        verify(executor, times(1)).traverseForestMulti(point,
                TestUtils.DUMMY_MULTI_VISITOR_FACTORY, accumulator, finisher);
    }

    @Test
    public void testTranverseForestMultiBinaryAccumulatorInvalid() {
        double[] point = {2.2, -1.1};
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(0.0);
                });

        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(null, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () ->
                forest.traverseForestMulti(new double[] {2.2, -1.1, 3.3}, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, (BinaryOperator<Double>) null, finisher));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestMultiCollector() {
        double[] point = {2.2, -1.1};

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTreeMulti(aryEq(point), any())).thenReturn(0.0);
                });

        forest.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR);
        verify(executor, times(1)).traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTranverseForestCollectorMultiInvalid() {
        double[] point = {2.2, -1.1};

        treeUpdaters.stream()
                .map(TreeUpdater::getTree)
                .forEach(tree -> {
                    when(tree.traverseTree(aryEq(point), any())).thenReturn(0.0);
                });

        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(null, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () ->
                forest.traverseForestMulti(new double[] {2.2, -1.1, 3.3}, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class, () ->
                forest.traverseForestMulti(point, TestUtils.DUMMY_MULTI_VISITOR_FACTORY, null));
    }

    @Test
    public void testGetAnomalyScore() {
        double[] point = {1.2, -3.4};

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            RandomCutTree tree = treeUpdaters.get(i).getTree();
            double treeResult = Math.random();
            when(tree.traverseTree(aryEq(point), any(AnomalyScoreVisitor.class))).thenReturn(treeResult);

            Node root = mock(Node.class);
            when(root.getMass()).thenReturn(256);
            when(tree.getRoot()).thenReturn(root);

            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;
        assertEquals(expectedResult, forest.getAnomalyScore(point), EPSILON);
    }

    @Test
    public void testGetApproximateAnomalyScore() {
        double[] point = {1.2, -3.4};

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getApproximateAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                RandomCutForest.DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED,
                numberOfTrees);

        for (int i = 0; i < numberOfTrees; i++) {
            RandomCutTree tree = treeUpdaters.get(i).getTree();
            double treeResult = Math.random();
            when(tree.traverseTree(aryEq(point), any(AnomalyScoreVisitor.class))).thenReturn(treeResult);

            Node root = mock(Node.class);
            when(root.getMass()).thenReturn(256);
            when(tree.getRoot()).thenReturn(root);

            if (!accumulator.isConverged()) {
                accumulator.accept(treeResult);
            }
        }

        double expectedResult = accumulator.getAccumulatedValue() / accumulator.getValuesAccepted();
        assertEquals(expectedResult, forest.getApproximateAnomalyScore(point), EPSILON);
    }

    @Test
    public void testGetAnomalyAttribution() {
        double[] point = {1.2, -3.4};

        assertFalse(forest.isOutputReady());
        DiVector zero = new DiVector(dimensions);
        DiVector result = forest.getAnomalyAttribution(point);
        assertArrayEquals(zero.high, result.high);
        assertArrayEquals(zero.low, result.low);

        doReturn(true).when(forest).isOutputReady();
        DiVector expectedResult = new DiVector(dimensions);

        for (int i = 0; i < numberOfTrees; i++) {
            DiVector treeResult = new DiVector(dimensions);
            for (int j = 0; j < dimensions; j++) {
                treeResult.high[j] = Math.random();
                treeResult.low[j] = Math.random();
            }

            RandomCutTree tree = treeUpdaters.get(i).getTree();
            when(tree.traverseTree(aryEq(point), any(AnomalyAttributionVisitor.class))).thenReturn(treeResult);

            Node root = mock(Node.class);
            when(root.getMass()).thenReturn(256);
            when(tree.getRoot()).thenReturn(root);

            DiVector.addToLeft(expectedResult, treeResult);
        }

        expectedResult = expectedResult.scale(1.0 / numberOfTrees);
        result = forest.getAnomalyAttribution(point);
        assertArrayEquals(expectedResult.high, result.high, EPSILON);
        assertArrayEquals(expectedResult.low, result.low, EPSILON);
    }

    @Test
    public void testGetApproximateAnomalyAttribution() {
        double[] point = {1.2, -3.4};
        DiVector zero = new DiVector(dimensions);
        DiVector result = forest.getApproximateAnomalyAttribution(point);

        assertFalse(forest.isOutputReady());
        assertArrayEquals(zero.high, result.high, EPSILON);
        assertArrayEquals(zero.low, result.low, EPSILON);

        doReturn(true).when(forest).isOutputReady();

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(
                dimensions,
                RandomCutForest.DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED,
                numberOfTrees);

        for (int i = 0; i < numberOfTrees; i++) {
            RandomCutTree tree = treeUpdaters.get(i).getTree();
            DiVector treeResult = new DiVector(dimensions);

            for (int j = 0; j < dimensions; j++) {
                treeResult.high[j] = Math.random();
                treeResult.low[j] = Math.random();
            }

            when(tree.traverseTree(aryEq(point), any(AnomalyAttributionVisitor.class))).thenReturn(treeResult);

            Node root = mock(Node.class);
            when(root.getMass()).thenReturn(256);
            when(tree.getRoot()).thenReturn(root);

            if (!accumulator.isConverged()) {
                accumulator.accept(treeResult);
            }
        }

        DiVector expectedResult = accumulator.getAccumulatedValue().scale(1.0 / accumulator.getValuesAccepted());
        result = forest.getApproximateAnomalyAttribution(point);
        assertArrayEquals(expectedResult.high, result.high, EPSILON);
        assertArrayEquals(expectedResult.low, result.low, EPSILON);
    }

    @Test
    public void testGetSimpleDensity() {
        double[] point = {12.3, -45.6};
        DensityOutput zero = new DensityOutput(dimensions, sampleSize);
        assertFalse(forest.samplersFull());
        DensityOutput result = forest.getSimpleDensity(point);
        assertEquals(zero.getDensity(), result.getDensity(), EPSILON);

        doReturn(true).when(forest).samplersFull();
        List<InterpolationMeasure> intermediateResults = new ArrayList<>();

        for (int i = 0; i < numberOfTrees; i++) {
            InterpolationMeasure treeResult = new InterpolationMeasure(dimensions, sampleSize);
            for (int j = 0; j < dimensions; j++) {
                treeResult.measure.high[j] = Math.random();
                treeResult.measure.low[j] = Math.random();
                treeResult.distances.high[j] = Math.random();
                treeResult.distances.low[j] = Math.random();
                treeResult.probMass.high[j] = Math.random();
                treeResult.probMass.low[j] = Math.random();
            }

            RandomCutTree tree = treeUpdaters.get(i).getTree();
            when(tree.traverseTree(aryEq(point), any(SimpleInterpolationVisitor.class))).thenReturn(treeResult);
            intermediateResults.add(treeResult);
        }

        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions, sampleSize, numberOfTrees);
        DensityOutput expectedResult = new DensityOutput(intermediateResults.stream().collect(collector));
        result = forest.getSimpleDensity(point);
        assertEquals(expectedResult.getDensity(), result.getDensity(), EPSILON);
    }

    @Test
    public void testImputeMissingValuesInvalid() {
        double[] point = {12.3, -45.6};
        int numberOfMissingValues = 1;
        int[] missingIndexes = {0, 1};

        assertThrows(IllegalArgumentException.class,
                () -> forest.imputeMissingValues(point, -1, missingIndexes));

        assertThrows(NullPointerException.class,
                () -> forest.imputeMissingValues(point, numberOfMissingValues, null));

        assertThrows(NullPointerException.class,
                () -> forest.imputeMissingValues(null, numberOfMissingValues, missingIndexes));

        int invalidNumberOfMissingValues = 99;
        assertThrows(IllegalArgumentException.class,
                () -> forest.imputeMissingValues(point, invalidNumberOfMissingValues, missingIndexes));
    }

    @Test
    public void testImputeMissingValuesWithNoMissingValues() {
        double[] point = {12.3, -45.6};
        int[] missingIndexes = {1, 1000}; // second value doesn't matter since numberOfMissingValues is 1o

        double[] result = forest.imputeMissingValues(point, 0, missingIndexes);
        assertArrayEquals(point, result);
    }

    @Test
    public void testImputMissingValuesWithOutputNotReady() {
        double[] point = {12.3, -45.6};
        int numberOfMissingValues = 1;
        int[] missingIndexes = {1, 1000}; // second value doesn't matter since numberOfMissingValues is 1o

        assertFalse(forest.isOutputReady());
        double[] zero = new double[dimensions];
        assertArrayEquals(zero, forest.imputeMissingValues(point, numberOfMissingValues, missingIndexes));
    }

    @Test
    public void testImputeMissingValuesWithSingleMissingIndex() {
        List<Double> returnValues = new ArrayList<>();
        for (int i = 0; i < numberOfTrees; i++) {
            returnValues.add((double) i);
        }
        double expectedResult = returnValues.get(numberOfTrees / 2);
        Collections.shuffle(returnValues);
        double[] point = {12.3, -45.6};

        int numberOfMissingValues = 1;
        int[] missingIndexes = {1, 999};

        for (int i = 0; i < numberOfTrees; i++) {
            RandomCutTree tree = treeUpdaters.get(i).getTree();
            double[] treeResult = Arrays.copyOf(point, point.length);
            treeResult[missingIndexes[0]] = returnValues.get(i);
            when(tree.traverseTreeMulti(aryEq(point), any(ImputeVisitor.class))).thenReturn(treeResult);
        }

        doReturn(true).when(forest).isOutputReady();
        double[] result = forest.imputeMissingValues(point, numberOfMissingValues, missingIndexes);

        for (int j = 0; j < dimensions; j++) {
            if (j == missingIndexes[0]) {
                assertEquals(expectedResult, result[j]);
            } else {
                assertEquals(point[j], result[j]);
            }
        }
    }

    @Test
    public void testImputeMissingValuesWithMultipleMissingIndexes() {
        double[] point = {12.3, -45.6};
        List<Double> anomalyScores = new ArrayList<>();

        for (int i = 0; i < numberOfTrees; i++) {
            anomalyScores.add((double) i);
        }

        double selectScore = anomalyScores.get(numberOfTrees / 4); // 25th percentile score
        Collections.shuffle(anomalyScores);

        int numberOfMissingValues = 2;
        int[] missingIndexes = {1, 0};
        double[] expectedResult = null;

        for (int i = 0; i < numberOfTrees; i++) {
            RandomCutTree tree = treeUpdaters.get(i).getTree();
            double[] treeResult = {Math.random(), Math.random()};
            when(tree.traverseTreeMulti(aryEq(point), any(ImputeVisitor.class))).thenReturn(treeResult);

            double anomalyScore = anomalyScores.get(i);
            doReturn(anomalyScore).when(forest).getAnomalyScore(aryEq(treeResult));
            if (anomalyScore == selectScore) {
                expectedResult = treeResult;
            }
        }

        doReturn(true).when(forest).isOutputReady();
        double[] result = forest.imputeMissingValues(point, numberOfMissingValues, missingIndexes);

        assertArrayEquals(expectedResult, result);
    }

    @Test
    public void testGetNearNeighborInSample() {
        List<Long> indexes1 = new ArrayList<>();
        indexes1.add(1L);
        indexes1.add(3L);

        List<Long> indexes2 = new ArrayList<>();
        indexes2.add(2L);
        indexes2.add(4L);

        List<Long> indexes4 = new ArrayList<>();
        indexes4.add(1L);
        indexes4.add(3L);

        List<Long> indexes5 = new ArrayList<>();
        indexes5.add(2L);
        indexes5.add(4L);

        Neighbor neighbor1 = new Neighbor(new double[]{1, 2}, 5, indexes1);
        RandomCutTree tree1 = mock(RandomCutTree.class);
        when(tree1.traverseTree(any(), any())).thenReturn(Optional.of(neighbor1));

        Neighbor neighbor2 = new Neighbor(new double[]{1, 2}, 5, indexes2);
        RandomCutTree tree2 = mock(RandomCutTree.class);
        when(tree2.traverseTree(any(), any())).thenReturn(Optional.of(neighbor2));

        RandomCutTree tree3 = mock(RandomCutTree.class);
        when(tree3.traverseTree(any(), any())).thenReturn(Optional.empty());

        Neighbor neighbor4 = new Neighbor(new double[]{2, 3}, 4, indexes4);
        RandomCutTree tree4 = mock(RandomCutTree.class);
        when(tree4.traverseTree(any(), any())).thenReturn(Optional.of(neighbor4));

        Neighbor neighbor5 = new Neighbor(new double[]{2, 3}, 4, indexes5);
        RandomCutTree tree5 = mock(RandomCutTree.class);
        when(tree5.traverseTree(any(), any())).thenReturn(Optional.of(neighbor5));

        RandomCutForest forest = spy(RandomCutForest.defaultForest(2));

        TreeUpdater treeUpdater1 = mock(TreeUpdater.class);
        when(treeUpdater1.getTree()).thenReturn(tree1);

        TreeUpdater treeUpdater2 = mock(TreeUpdater.class);
        when(treeUpdater2.getTree()).thenReturn(tree2);

        TreeUpdater treeUpdater3 = mock(TreeUpdater.class);
        when(treeUpdater3.getTree()).thenReturn(tree3);

        TreeUpdater treeUpdater4 = mock(TreeUpdater.class);
        when(treeUpdater4.getTree()).thenReturn(tree4);

        TreeUpdater treeUpdater5 = mock(TreeUpdater.class);
        when(treeUpdater5.getTree()).thenReturn(tree5);

        ArrayList<TreeUpdater> treeUpdaters = new ArrayList<>();
        treeUpdaters.add(treeUpdater1);
        treeUpdaters.add(treeUpdater2);
        treeUpdaters.add(treeUpdater3);
        treeUpdaters.add(treeUpdater4);
        treeUpdaters.add(treeUpdater5);

        AbstractForestTraversalExecutor executor = new SequentialForestTraversalExecutor(treeUpdaters);

        Whitebox.setInternalState(forest, "storeSequenceIndexesEnabled", true);
        Whitebox.setInternalState(forest, "executor", executor);

        doReturn(true).when(forest).isOutputReady();
        List<Neighbor> neighbors = forest.getNearNeighborsInSample(new double[]{0, 0}, 5);

        List<Long> expectedIndexes = Arrays.asList(1L, 2L, 3L, 4L);
        assertEquals(2, neighbors.size());
        assertTrue(neighbors.get(0).point[0] == 2 && neighbors.get(0).point[1] == 3);
        assertEquals(4, neighbors.get(0).distance);
        assertEquals(4, neighbors.get(0).sequenceIndexes.size());
        assertThat(neighbors.get(0).sequenceIndexes, is(expectedIndexes));

        assertTrue(neighbors.get(1).point[0] == 1 && neighbors.get(1).point[1] == 2);
        assertEquals(5, neighbors.get(1).distance );
        assertEquals(4, neighbors.get(1).sequenceIndexes.size());
        assertThat(neighbors.get(1).sequenceIndexes, is(expectedIndexes));
    }

    @Test
    public void testUpdateOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder()
            .dimensions(1)
            .numberOfTrees(1)
            .sampleSize(2)
            .lambda(0.5)
            .randomSeed(0)
            .parallelExecutionEnabled(false);

       RandomCutForest forest = forestBuilder.build();
       double[][] data = new double[][] {
           {48.08}
           ,{48.08000000000001}
       };

       for (int i = 0; i < 20000; i++) {
           forest.update(data[i % data.length]);
       }
    }
}
