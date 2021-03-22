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
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.powermock.reflect.Whitebox;

import com.amazon.randomcutforest.executor.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.executor.AbstractForestUpdateExecutor;
import com.amazon.randomcutforest.executor.IUpdateCoordinator;
import com.amazon.randomcutforest.executor.PassThroughCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.executor.SequentialForestTraversalExecutor;
import com.amazon.randomcutforest.executor.SequentialForestUpdateExecutor;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDiVectorAccumulator;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDoubleAccumulator;
import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.ShingleBuilder;

public class RandomCutForestTest {

    private int dimensions;
    private int sampleSize;
    private int numberOfTrees;
    private ComponentList<double[], double[]> components;
    private AbstractForestTraversalExecutor traversalExecutor;
    private IUpdateCoordinator<double[], double[]> updateCoordinator;
    private AbstractForestUpdateExecutor<double[], double[]> updateExecutor;
    private RandomCutForest forest;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        sampleSize = 256;
        numberOfTrees = 10;

        components = new ComponentList<>();
        for (int i = 0; i < numberOfTrees; i++) {
            SimpleStreamSampler<double[]> sampler = mock(SimpleStreamSampler.class);
            RandomCutTree tree = mock(RandomCutTree.class);
            components.add(spy(new SamplerPlusTree<>(sampler, tree)));

        }
        updateCoordinator = spy(new PassThroughCoordinator());
        traversalExecutor = spy(new SequentialForestTraversalExecutor(components));
        updateExecutor = spy(new SequentialForestUpdateExecutor<>(updateCoordinator, components));

        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize);
        forest = spy(new RandomCutForest(builder, updateCoordinator, components, builder.getRandom()));

        Whitebox.setInternalState(forest, "traversalExecutor", traversalExecutor);
        Whitebox.setInternalState(forest, "updateExecutor", updateExecutor);
    }

    @Test
    public void testUpdate() {
        double[] point = { 2.2, -1.1 };
        forest.update(point);
        verify(updateExecutor, times(1)).update(point);
    }

    @Test
    public void testUpdateInvalid() {
        assertThrows(NullPointerException.class, () -> forest.update(null));
        assertThrows(IllegalArgumentException.class, () -> forest.update(new double[] { 1.2, 3.4, -5.6 }));
    }

    @Test
    public void testTraverseForestBinaryAccumulator() {
        double[] point = { 2.2, -1.1 };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> {
            doReturn(0.0).when(c).traverse(aryEq(point), any(Function.class));
        });

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator,
                finisher);
    }

    @Test
    public void testTranverseForestBinaryAccumulatorInvalid() {
        double[] point = { 2.2, -1.1 };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> {
            when(c.traverse(aryEq(point), any())).thenReturn(0.0);
        });

        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(null, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new double[] { 2.2, -1.1, 3.3 },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, (BinaryOperator<Double>) null, finisher));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestCollector() {
        double[] point = { 2.2, -1.1 };

        components.forEach(c -> {
            doReturn(0.0).when(c).traverse(aryEq(point), any(Function.class));
        });

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTranverseForestCollectorInvalid() {
        double[] point = { 2.2, -1.1 };

        components.forEach(c -> {
            when(c.traverse(aryEq(point), any())).thenReturn(0.0);
        });

        assertThrows(NullPointerException.class, () -> forest.traverseForest(null,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new double[] { 2.2, -1.1, 3.3 },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, null));
    }

    @Test
    public void testTraverseForestConverging() {
        double[] point = new double[] { 1.2, -3.4 };

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        components.forEach(c -> {
            doReturn(0.0).when(c).traverse(aryEq(point), any(Function.class));
        });

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator,
                finisher);
    }

    @Test
    public void testTraverseForestConvergingInvalid() {
        double[] point = new double[] { 1.2, -3.4 };

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        components.forEach(c -> {
            when(c.traverse(aryEq(point), any())).thenReturn(0.0);
        });

        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(null, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new double[] { 1.2, -3.4, 5.6 },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, (ConvergingAccumulator<Double>) null, finisher));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void traverseForestMultiBinaryAccumulator() {
        double[] point = { 2.2, -1.1 };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> {
            doReturn(0.0).when(c).traverseMulti(aryEq(point), any(Function.class));
        });

        forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                accumulator, finisher);
    }

    @Test
    public void testTranverseForestMultiBinaryAccumulatorInvalid() {
        double[] point = { 2.2, -1.1 };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> {
            when(c.traverseMulti(aryEq(point), any())).thenReturn(0.0);
        });

        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(null,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForestMulti(new double[] { 2.2, -1.1, 3.3 },
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(point,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, (BinaryOperator<Double>) null, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(point,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestMultiCollector() {
        double[] point = { 2.2, -1.1 };

        components.forEach(c -> {
            doReturn(0.0).when(c).traverseMulti(aryEq(point), any(Function.class));
        });

        forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
        verify(traversalExecutor, times(1)).traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTranverseForestCollectorMultiInvalid() {
        double[] point = { 2.2, -1.1 };

        components.forEach(c -> {
            when(c.traverse(aryEq(point), any())).thenReturn(0.0);
        });

        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(null,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForestMulti(new double[] { 2.2, -1.1, 3.3 },
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForestMulti(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, null));
    }

    @Test
    public void testGetAnomalyScore() {
        double[] point = { 1.2, -3.4 };

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            double treeResult = Math.random();
            when(tree.traverse(aryEq(point), any(Function.class))).thenReturn(treeResult);

            when(tree.getMass()).thenReturn(256);

            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;
        assertEquals(expectedResult, forest.getAnomalyScore(point), EPSILON);
    }

    @Test
    public void testGetApproximateAnomalyScore() {
        double[] point = { 1.2, -3.4 };

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getApproximateAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                RandomCutForest.DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            double treeResult = Math.random();
            when(tree.traverse(aryEq(point), any(Function.class))).thenReturn(treeResult);

            when(tree.getMass()).thenReturn(256);

            if (!accumulator.isConverged()) {
                accumulator.accept(treeResult);
            }
        }

        double expectedResult = accumulator.getAccumulatedValue() / accumulator.getValuesAccepted();
        assertEquals(expectedResult, forest.getApproximateAnomalyScore(point), EPSILON);
    }

    @Test
    public void testGetAnomalyAttribution() {
        double[] point = { 1.2, -3.4 };

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

            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            when(tree.traverse(aryEq(point), any(Function.class))).thenReturn(treeResult);

            when(tree.getMass()).thenReturn(256);

            DiVector.addToLeft(expectedResult, treeResult);
        }

        expectedResult = expectedResult.scale(1.0 / numberOfTrees);
        result = forest.getAnomalyAttribution(point);
        assertArrayEquals(expectedResult.high, result.high, EPSILON);
        assertArrayEquals(expectedResult.low, result.low, EPSILON);
    }

    @Test
    public void testGetApproximateAnomalyAttribution() {
        double[] point = { 1.2, -3.4 };
        DiVector zero = new DiVector(dimensions);
        DiVector result = forest.getApproximateAnomalyAttribution(point);

        assertFalse(forest.isOutputReady());
        assertArrayEquals(zero.high, result.high, EPSILON);
        assertArrayEquals(zero.low, result.low, EPSILON);

        doReturn(true).when(forest).isOutputReady();

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions,
                RandomCutForest.DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            DiVector treeResult = new DiVector(dimensions);

            for (int j = 0; j < dimensions; j++) {
                treeResult.high[j] = Math.random();
                treeResult.low[j] = Math.random();
            }

            when(tree.traverse(aryEq(point), any(Function.class))).thenReturn(treeResult);

            when(tree.getMass()).thenReturn(256);

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
        double[] point = { 12.3, -45.6 };
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

            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            when(tree.traverse(aryEq(point), any(Function.class))).thenReturn(treeResult);
            intermediateResults.add(treeResult);
        }

        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions,
                sampleSize, numberOfTrees);
        DensityOutput expectedResult = new DensityOutput(intermediateResults.stream().collect(collector));
        result = forest.getSimpleDensity(point);
        assertEquals(expectedResult.getDensity(), result.getDensity(), EPSILON);
    }

    @Test
    public void testImputeMissingValuesInvalid() {
        double[] point = { 12.3, -45.6 };
        int numberOfMissingValues = 1;
        int[] missingIndexes = { 0, 1 };

        assertThrows(IllegalArgumentException.class, () -> forest.imputeMissingValues(point, -1, missingIndexes));

        assertThrows(NullPointerException.class, () -> forest.imputeMissingValues(point, numberOfMissingValues, null));

        assertThrows(NullPointerException.class,
                () -> forest.imputeMissingValues(null, numberOfMissingValues, missingIndexes));

        int invalidNumberOfMissingValues = 99;
        assertThrows(IllegalArgumentException.class,
                () -> forest.imputeMissingValues(point, invalidNumberOfMissingValues, missingIndexes));
    }

    @Test
    public void testImputeMissingValuesWithNoMissingValues() {
        double[] point = { 12.3, -45.6 };
        int[] missingIndexes = { 1, 1000 }; // second value doesn't matter since numberOfMissingValues is 1o

        double[] result = forest.imputeMissingValues(point, 0, missingIndexes);
        assertArrayEquals(point, result);
    }

    @Test
    public void testImputMissingValuesWithOutputNotReady() {
        double[] point = { 12.3, -45.6 };
        int numberOfMissingValues = 1;
        int[] missingIndexes = { 1, 1000 }; // second value doesn't matter since numberOfMissingValues is 1o

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
        double[] point = { 12.3, -45.6 };

        int numberOfMissingValues = 1;
        int[] missingIndexes = { 1, 999 };

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            double[] treeResult = Arrays.copyOf(point, point.length);
            treeResult[missingIndexes[0]] = returnValues.get(i);
            when(tree.traverseMulti(aryEq(point), any(Function.class))).thenReturn(treeResult);
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
        double[] point = { 12.3, -45.6 };
        List<Double> anomalyScores = new ArrayList<>();

        for (int i = 0; i < numberOfTrees; i++) {
            anomalyScores.add((double) i);
        }

        double selectScore = anomalyScores.get(numberOfTrees / 4); // 25th percentile score
        Collections.shuffle(anomalyScores);

        int numberOfMissingValues = 2;
        int[] missingIndexes = { 1, 0 };
        double[] expectedResult = null;

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<double[], double[]> component = (SamplerPlusTree<double[], double[]>) components.get(i);
            ITree<double[], double[]> tree = component.getTree();
            double[] treeResult = { Math.random(), Math.random() };
            when(tree.traverseMulti(aryEq(point), any(Function.class))).thenReturn(treeResult);

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
    public void testExtrapolateBasic() {
        doNothing().when(forest).extrapolateBasicCyclic(any(double[].class), anyInt(), anyInt(), anyInt(),
                any(double[].class), any(int[].class));
        doNothing().when(forest).extrapolateBasicSliding(any(double[].class), anyInt(), anyInt(), any(double[].class),
                any(int[].class));

        double[] point = new double[] { 2.0, -3.0 };
        int horizon = 2;
        int blockSize = 1;
        boolean cyclic = true;
        int shingleIndex = 1;

        forest.extrapolateBasic(point, horizon, blockSize, cyclic, shingleIndex);
        verify(forest).extrapolateBasicCyclic(any(double[].class), eq(horizon), eq(blockSize), eq(shingleIndex),
                any(double[].class), any(int[].class));

        forest.extrapolateBasic(point, horizon, blockSize, cyclic);
        verify(forest).extrapolateBasicCyclic(any(double[].class), eq(horizon), eq(blockSize), eq(0),
                any(double[].class), any(int[].class));

        cyclic = false;
        forest.extrapolateBasic(point, horizon, blockSize, cyclic, shingleIndex);
        forest.extrapolateBasic(point, horizon, blockSize, cyclic);
        verify(forest, times(2)).extrapolateBasicSliding(any(double[].class), eq(horizon), eq(blockSize),
                any(double[].class), any(int[].class));
    }

    @Test
    public void testExtrapolateBasicInvalid() {
        double[] point = new double[] { 2.0, -3.0 };
        int horizon = 2;
        int blockSize = 1;
        boolean cyclic = true;
        int shingleIndex = 1;

        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(point, horizon, -10, cyclic, shingleIndex));
        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(point, horizon, 0, cyclic, shingleIndex));
        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(point, horizon, dimensions, cyclic, shingleIndex));
        assertThrows(IllegalArgumentException.class,
                () -> forest.extrapolateBasic(point, horizon, dimensions * 2, cyclic, shingleIndex));
        assertThrows(NullPointerException.class,
                () -> forest.extrapolateBasic(null, horizon, blockSize, cyclic, shingleIndex));

        RandomCutForest f = RandomCutForest.defaultForest(20);
        double[] p = new double[20];

        // dimensions not divisible by blockSize
        assertThrows(IllegalArgumentException.class, () -> f.extrapolateBasic(p, horizon, 7, cyclic, shingleIndex));

        // invalid shingle index values
        assertThrows(IllegalArgumentException.class, () -> f.extrapolateBasic(point, horizon, 5, cyclic, -1));
        assertThrows(IllegalArgumentException.class, () -> f.extrapolateBasic(point, horizon, 5, cyclic, 4));
        assertThrows(IllegalArgumentException.class, () -> f.extrapolateBasic(point, horizon, 4, cyclic, 44));
    }

    @Test
    public void testExrapolateBasicWithShingleBuilder() {
        doNothing().when(forest).extrapolateBasicCyclic(any(double[].class), anyInt(), anyInt(), anyInt(),
                any(double[].class), any(int[].class));
        doNothing().when(forest).extrapolateBasicSliding(any(double[].class), anyInt(), anyInt(), any(double[].class),
                any(int[].class));

        ShingleBuilder shingleBuilder = new ShingleBuilder(1, 2, true);
        int horizon = 3;

        forest.extrapolateBasic(shingleBuilder, horizon);
        verify(forest, times(1)).extrapolateBasicCyclic(any(double[].class), eq(horizon), eq(1), eq(0),
                any(double[].class), any(int[].class));

        shingleBuilder = new ShingleBuilder(1, 2, false);
        forest.extrapolateBasic(shingleBuilder, horizon);
        verify(forest, times(1)).extrapolateBasicSliding(any(double[].class), eq(horizon), eq(1), any(double[].class),
                any(int[].class));
    }

    @Test
    public void testExtrapolateBasicSliding() {
        int horizon = 3;
        int blockSize = 2;
        double[] result = new double[dimensions * horizon];
        double[] queryPoint = new double[] { 1.0, -2.0 };
        int[] missingIndexes = new int[blockSize];

        doReturn(new double[] { 2.0, -3.0 }).doReturn(new double[] { 4.0, -5.0 }).doReturn(new double[] { 6.0, -7.0 })
                .when(forest).imputeMissingValues(aryEq(queryPoint), eq(blockSize), any(int[].class));

        forest.extrapolateBasicSliding(result, horizon, blockSize, queryPoint, missingIndexes);

        double[] expectedResult = new double[] { 2.0, -3.0, 4.0, -5.0, 6.0, -7.0 };
        assertArrayEquals(expectedResult, result);
    }

    @Test
    public void testExtrapolateBasicCyclic() {
        int horizon = 3;
        int blockSize = 2;
        double[] result = new double[dimensions * horizon];
        int shingleIndex = 1;
        double[] queryPoint = new double[] { 1.0, -2.0 };
        int[] missingIndexes = new int[blockSize];

        doReturn(new double[] { 2.0, -3.0 }).doReturn(new double[] { 4.0, -5.0 }).doReturn(new double[] { 6.0, -7.0 })
                .when(forest).imputeMissingValues(aryEq(queryPoint), eq(blockSize), any(int[].class));

        forest.extrapolateBasicCyclic(result, horizon, blockSize, shingleIndex, queryPoint, missingIndexes);

        double[] expectedResult = new double[] { -3.0, 2.0, -5.0, 4.0, -7.0, 6.0 };
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

        Neighbor neighbor1 = new Neighbor(new double[] { 1, 2 }, 5, indexes1);
        when(((SamplerPlusTree<?, ?>) components.get(0)).getTree().traverse(any(double[].class), any(Function.class)))
                .thenReturn(Optional.of(neighbor1));

        Neighbor neighbor2 = new Neighbor(new double[] { 1, 2 }, 5, indexes2);
        when(((SamplerPlusTree<?, ?>) components.get(1)).getTree().traverse(any(double[].class), any(Function.class)))
                .thenReturn(Optional.of(neighbor2));

        when(((SamplerPlusTree<?, ?>) components.get(2)).getTree().traverse(any(double[].class), any(Function.class)))
                .thenReturn(Optional.empty());

        Neighbor neighbor4 = new Neighbor(new double[] { 2, 3 }, 4, indexes4);
        when(((SamplerPlusTree<?, ?>) components.get(3)).getTree().traverse(any(double[].class), any(Function.class)))
                .thenReturn(Optional.of(neighbor4));

        Neighbor neighbor5 = new Neighbor(new double[] { 2, 3 }, 4, indexes5);
        when(((SamplerPlusTree<?, ?>) components.get(4)).getTree().traverse(any(double[].class), any(Function.class)))
                .thenReturn(Optional.of(neighbor5));

        for (int i = 5; i < components.size(); i++) {
            when(((SamplerPlusTree<?, ?>) components.get(i)).getTree().traverse(any(double[].class),
                    any(Function.class))).thenReturn(Optional.empty());
        }

        Whitebox.setInternalState(forest, "storeSequenceIndexesEnabled", true);

        doReturn(true).when(forest).isOutputReady();
        List<Neighbor> neighbors = forest.getNearNeighborsInSample(new double[] { 0, 0 }, 5);

        List<Long> expectedIndexes = Arrays.asList(1L, 2L, 3L, 4L);
        assertEquals(2, neighbors.size());
        assertTrue(neighbors.get(0).point[0] == 2 && neighbors.get(0).point[1] == 3);
        assertEquals(4, neighbors.get(0).distance);
        assertEquals(4, neighbors.get(0).sequenceIndexes.size());
        assertThat(neighbors.get(0).sequenceIndexes, is(expectedIndexes));

        assertTrue(neighbors.get(1).point[0] == 1 && neighbors.get(1).point[1] == 2);
        assertEquals(5, neighbors.get(1).distance);
        assertEquals(4, neighbors.get(1).sequenceIndexes.size());
        assertThat(neighbors.get(1).sequenceIndexes, is(expectedIndexes));
    }

    @Test
    public void testGetNearNeighborsInSampleBeforeOutputReady() {
        assertFalse(forest.isOutputReady());
        assertTrue(forest.getNearNeighborsInSample(new double[] { 0.1, 0.2 }, 5.0).isEmpty());
    }

    @Test
    public void testGetNearNeighborsInSampleNoDistanceThreshold() {
        forest.getNearNeighborsInSample(new double[] { 0.1, 0.2 });
        verify(forest, times(1)).getNearNeighborsInSample(aryEq(new double[] { 0.1, 0.2 }),
                eq(Double.POSITIVE_INFINITY));
    }

    @Test
    public void testGetNearNeighborsInSampleInvalid() {
        assertThrows(NullPointerException.class, () -> forest.getNearNeighborsInSample(null, 101.1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getNearNeighborsInSample(new double[] { 1.1, 2.2 }, -101.1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getNearNeighborsInSample(new double[] { 1.1, 2.2 }, 0.0));
    }

    @Test
    public void testUpdateOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder().dimensions(1).numberOfTrees(1).sampleSize(3)
                .lambda(0.5).randomSeed(0).parallelExecutionEnabled(false);

        RandomCutForest forest = forestBuilder.build();
        double[][] data = new double[][] { { 48.08 }, { 48.08000000000001 } };

        for (int i = 0; i < 20000; i++) {
            forest.update(data[i % data.length]);
        }
    }

    @Test
    public void testSamplersFull() {
        long totalUpdates = sampleSize / 2;
        when(updateCoordinator.getTotalUpdates()).thenReturn(totalUpdates);
        assertFalse(forest.samplersFull());

        totalUpdates = sampleSize;
        when(updateCoordinator.getTotalUpdates()).thenReturn(totalUpdates);
        assertTrue(forest.samplersFull());

        totalUpdates = sampleSize * 10;
        when(updateCoordinator.getTotalUpdates()).thenReturn(totalUpdates);
        assertTrue(forest.samplersFull());
    }

    @Test
    public void testGetTotalUpdates() {
        long totalUpdates = 987654321L;
        when(updateCoordinator.getTotalUpdates()).thenReturn(totalUpdates);
        assertEquals(totalUpdates, forest.getTotalUpdates());
    }
}
