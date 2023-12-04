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

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalMatchers.aryEq;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.powermock.reflect.Whitebox;

import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.executor.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.executor.AbstractForestUpdateExecutor;
import com.amazon.randomcutforest.executor.IStateCoordinator;
import com.amazon.randomcutforest.executor.PointStoreCoordinator;
import com.amazon.randomcutforest.executor.SamplerPlusTree;
import com.amazon.randomcutforest.executor.SequentialForestTraversalExecutor;
import com.amazon.randomcutforest.executor.SequentialForestUpdateExecutor;
import com.amazon.randomcutforest.returntypes.ConditionalTreeSample;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.InterpolationMeasure;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDiVectorAccumulator;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDoubleAccumulator;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.sampler.CompactSampler;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.amazon.randomcutforest.util.ShingleBuilder;

public class RandomCutForestTest {

    private int dimensions;
    private int sampleSize;
    private int numberOfTrees;
    private ComponentList<Integer, float[]> components;
    private AbstractForestTraversalExecutor traversalExecutor;
    private IStateCoordinator<Integer, float[]> updateCoordinator;
    private AbstractForestUpdateExecutor<Integer, float[]> updateExecutor;
    private RandomCutForest forest;

    @BeforeEach
    public void setUp() {
        dimensions = 2;
        sampleSize = 256;
        numberOfTrees = 10;

        components = new ComponentList<>();
        for (int i = 0; i < numberOfTrees; i++) {
            CompactSampler sampler = mock(CompactSampler.class);
            when(sampler.getCapacity()).thenReturn(sampleSize);
            RandomCutTree tree = mock(RandomCutTree.class);
            components.add(spy(new SamplerPlusTree<>(sampler, tree)));

        }
        updateCoordinator = spy(
                new PointStoreCoordinator<>(new PointStore.Builder().dimensions(2).capacity(1).build()));
        traversalExecutor = spy(new SequentialForestTraversalExecutor(components));
        updateExecutor = spy(new SequentialForestUpdateExecutor<>(updateCoordinator, components));

        RandomCutForest.Builder<?> builder = RandomCutForest.builder().dimensions(dimensions)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize);
        forest = spy(new RandomCutForest(builder, updateCoordinator, components, builder.getRandom()));

        Whitebox.setInternalState(forest, "traversalExecutor", traversalExecutor);
        Whitebox.setInternalState(forest, "updateExecutor", updateExecutor);
    }

    @Test
    void checkOutput() {
        assertFalse(forest.isOutputReady());
        assertEquals(forest.getConditionalField(null, 1, new int[1], 1.0).size(), 0);
        assertEquals(forest.extrapolateBasic(new float[2], 1, 1, false)[0], 0);
        assertEquals(forest.getDynamicScore(new float[2], 1, null, null, null), 0);
        assertEquals(forest.getDynamicAttribution(new float[2], 1, null, null, null).getHighLowSum(), 0);
        assertEquals(forest.getDynamicSimulatedScore(new float[2], null, null, null, null), 0);
        assertEquals(forest.getApproximateDynamicScore(new float[2], 0.1, true, 1, null, null, null), 0);
        assertEquals(
                forest.getApproximateDynamicAttribution(new float[2], 0.1, true, 1, null, null, null).getHighLowSum(),
                0);
    }

    @Test
    void checkParameters() {
        assertThrows(IllegalArgumentException.class, () -> forest.getConditionalField(null, 0, null, 1));
        assertThrows(IllegalArgumentException.class, () -> forest.getConditionalField(null, 2, new int[1], 1));
        assertThrows(IllegalArgumentException.class, () -> forest.getConditionalField(null, 1, new int[1], -1));
        assertThrows(IllegalArgumentException.class, () -> forest.getConditionalField(null, 1, new int[1], 2));
        assertDoesNotThrow(() -> forest.getConditionalField(null, 1, new int[1], 1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getConditionalFieldSummary(new float[2], 0, new int[0], 1, 0, false, false, -1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getConditionalFieldSummary(new float[2], 0, new int[0], 1, 0, false, false, 2, 1));
        assertDoesNotThrow(
                () -> forest.getConditionalFieldSummary(new float[2], 0, new int[0], 1, 0, false, false, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> forest.setTimeDecay(-2));
        assertThrows(IllegalArgumentException.class, () -> forest.setBoundingBoxCacheFraction(-1));
        assertThrows(IllegalArgumentException.class, () -> forest.setBoundingBoxCacheFraction(2));
        assertThrows(IllegalArgumentException.class, () -> forest.getDynamicScore(new float[2], -1, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getApproximateDynamicScore(new float[2], 0.1, true, -1, null, null, null));
    }

    @Test
    public void testUpdate() {
        float[] point = { 2.2f, -1.1f };
        forest.update(point);
        verify(updateExecutor, times(1)).update(point);
        assertEquals(updateCoordinator.getStore().getCapacity(), 1);
    }

    @Test
    public void testUpdateShingled() {
        float[] point = { 2.2f, -1.1f };
        RandomCutForest newForest = RandomCutForest.builder().internalShinglingEnabled(true).dimensions(2).build();
        assertThrows(IllegalArgumentException.class, () -> newForest.update(point, 0L));
        assertDoesNotThrow(() -> newForest.update(point));
    }

    @Test
    public void testUpdateInvalid() {
        assertThrows(NullPointerException.class, () -> forest.update((double[]) null));
        assertThrows(NullPointerException.class, () -> forest.update((float[]) null));
        assertThrows(IllegalArgumentException.class, () -> forest.update(new double[] { 1.2, 3.4, -5.6 }));
        assertThrows(IllegalArgumentException.class, () -> forest.update(new float[3]));
        assertThrows(IllegalArgumentException.class, () -> forest.update(new float[3], 0l));
    }

    @Test
    public void testTraverseForestBinaryAccumulator() {
        float[] point = { 2.2f, -1.1f };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> doReturn(0.0).when(c).traverse(aryEq(point), any(VisitorFactory.class)));

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator,
                finisher);
    }

    @Test
    public void testTraverseForestBinaryAccumulatorInvalid() {
        float[] point = { 2.2f, -1.1f };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> when(c.traverse(aryEq(point), any())).thenReturn(0.0));

        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(null, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new float[] { 2.2f, -1.1f, 3.3f },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, (BinaryOperator<Double>) null, finisher));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestCollector() {
        float[] point = { 2.2f, -1.1f };

        components.forEach(c -> doReturn(0.0).when(c).traverse(aryEq(point), any(VisitorFactory.class)));

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTraverseForestCollectorInvalid() {
        float[] point = { 2.2f, -1.1f };

        components.forEach(c -> when(c.traverse(aryEq(point), any())).thenReturn(0.0));

        assertThrows(NullPointerException.class, () -> forest.traverseForest(null,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new float[] { 2.2f, -1.1f, 3.3f },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, null));
    }

    @Test
    public void testTraverseForestConverging() {
        float[] point = new float[] { 1.2f, -3.4f };

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        components.forEach(c -> doReturn(0.0).when(c).traverse(aryEq(point), any(VisitorFactory.class)));

        forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator,
                finisher);
    }

    @Test
    public void testTraverseForestConvergingInvalid() {
        float[] point = new float[] { 1.2f, -3.4f };

        int convergenceThreshold = numberOfTrees / 2;
        ConvergingAccumulator<Double> accumulator = TestUtils.convergeAfter(convergenceThreshold);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        components.forEach(c -> when(c.traverse(aryEq(point), any())).thenReturn(0.0));

        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(null, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForest(new float[] { 1.2f, -3.4f, 5.6f },
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForest(point,
                TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, (ConvergingAccumulator<Double>) null, finisher));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForest(point, TestUtils.DUMMY_GENERIC_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void traverseForestMultiBinaryAccumulator() {
        float[] point = { 2.2f, -1.1f };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> doReturn(0.0).when(c).traverseMulti(aryEq(point), any(MultiVisitorFactory.class)));

        forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher);
        verify(traversalExecutor, times(1)).traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                accumulator, finisher);
    }

    @Test
    public void testTraverseForestMultiBinaryAccumulatorInvalid() {
        float[] point = { 2.2f, -1.1f };
        BinaryOperator<Double> accumulator = Double::sum;
        Function<Double, Double> finisher = x -> x / numberOfTrees;

        components.forEach(c -> when(c.traverseMulti(aryEq(point), any())).thenReturn(0.0));

        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(null,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForestMulti(new float[] { 2.2f, -1.1f, 3.3f },
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(point, null, accumulator, finisher));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, null, finisher));
        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(point,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, accumulator, null));
    }

    @Test
    public void testTraverseForestMultiCollector() {
        float[] point = { 2.2f, -1.1f };

        components.forEach(c -> doReturn(0.0).when(c).traverseMulti(aryEq(point), any(MultiVisitorFactory.class)));

        forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
        verify(traversalExecutor, times(1)).traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY,
                TestUtils.SORTED_LIST_COLLECTOR);
    }

    @Test
    public void testTraverseForestCollectorMultiInvalid() {
        float[] point = { 2.2f, -1.1f };

        components.forEach(c -> when(c.traverse(aryEq(point), any())).thenReturn(0.0));

        assertThrows(NullPointerException.class, () -> forest.traverseForestMulti(null,
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(IllegalArgumentException.class, () -> forest.traverseForestMulti(new float[] { 2.2f, -1.1f, 3.3f },
                TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForestMulti(point, null, TestUtils.SORTED_LIST_COLLECTOR));
        assertThrows(NullPointerException.class,
                () -> forest.traverseForestMulti(point, TestUtils.DUMMY_GENERIC_MULTI_VISITOR_FACTORY, null));
    }

    @Test
    public void testGetAnomalyScore() {
        float[] point = { 1.2f, -3.4f };

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();
        double expectedResult = 0.0;

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<Integer, float[]> component = (SamplerPlusTree<Integer, float[]>) components.get(i);
            ITree<Integer, float[]> tree = component.getTree();
            double treeResult = Math.random();
            when(tree.traverse(aryEq(point), any(IVisitorFactory.class))).thenReturn(treeResult);

            when(tree.getMass()).thenReturn(256);

            expectedResult += treeResult;
        }

        expectedResult /= numberOfTrees;
        assertEquals(expectedResult, forest.getAnomalyScore(point), EPSILON);
    }

    @Test
    public void testGetApproximateAnomalyScore() {
        float[] point = { 1.2f, -3.4f };

        assertFalse(forest.isOutputReady());
        assertEquals(0.0, forest.getApproximateAnomalyScore(point));

        doReturn(true).when(forest).isOutputReady();

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(
                RandomCutForest.DEFAULT_APPROXIMATE_ANOMALY_SCORE_HIGH_IS_CRITICAL,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_PRECISION,
                RandomCutForest.DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        for (int i = 0; i < numberOfTrees; i++) {
            SamplerPlusTree<Integer, float[]> component = (SamplerPlusTree<Integer, float[]>) components.get(i);
            ITree<Integer, float[]> tree = component.getTree();
            double treeResult = Math.random();
            when(tree.traverse(aryEq(point), any(IVisitorFactory.class))).thenReturn(treeResult);

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
        float[] point = { 1.2f, -3.4f };

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

            SamplerPlusTree<Integer, float[]> component = (SamplerPlusTree<Integer, float[]>) components.get(i);
            ITree<Integer, float[]> tree = component.getTree();
            when(tree.traverse(aryEq(point), any(VisitorFactory.class))).thenReturn(treeResult);

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
        float[] point = { 1.2f, -3.4f };
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
            SamplerPlusTree<Integer, float[]> component = (SamplerPlusTree<Integer, float[]>) components.get(i);
            ITree<Integer, float[]> tree = component.getTree();
            DiVector treeResult = new DiVector(dimensions);

            for (int j = 0; j < dimensions; j++) {
                treeResult.high[j] = Math.random();
                treeResult.low[j] = Math.random();
            }

            when(tree.traverse(aryEq(point), any(VisitorFactory.class))).thenReturn(treeResult);

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
        float[] point = { 12.3f, -45.6f };
        DensityOutput zero = new DensityOutput(dimensions, sampleSize);
        assertFalse(forest.isOutputReady());
        DensityOutput result = forest.getSimpleDensity(point);
        assertEquals(zero.getDensity(), result.getDensity(), EPSILON);

        doReturn(true).when(forest).isOutputReady();
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

            SamplerPlusTree<Integer, float[]> component = (SamplerPlusTree<Integer, float[]>) components.get(i);
            ITree<Integer, float[]> tree = component.getTree();
            when(tree.traverse(aryEq(point), any(VisitorFactory.class))).thenReturn(treeResult);
            intermediateResults.add(treeResult);
        }

        Collector<InterpolationMeasure, ?, InterpolationMeasure> collector = InterpolationMeasure.collector(dimensions,
                0, numberOfTrees);
        DensityOutput expectedResult = new DensityOutput(intermediateResults.stream().collect(collector));
        result = forest.getSimpleDensity(point);
        assertEquals(expectedResult.getDensity(), result.getDensity(), EPSILON);
    }

    @Test
    public void testImputeMissingValuesInvalid() {
        float[] point = { 12.3f, -45.6f };
        int numberOfMissingValues = 1;
        int[] missingIndexes = { 0, 1 };

        assertThrows(IllegalArgumentException.class, () -> forest.imputeMissingValues(point, -1, missingIndexes));

        assertThrows(NullPointerException.class, () -> forest.imputeMissingValues(point, numberOfMissingValues, null));

        assertThrows(IllegalArgumentException.class,
                () -> forest.imputeMissingValues((float[]) null, numberOfMissingValues, missingIndexes));

        int invalidNumberOfMissingValues = 99;
        assertThrows(IllegalArgumentException.class,
                () -> forest.imputeMissingValues(point, invalidNumberOfMissingValues, missingIndexes));
    }

    @Test
    public void testImputeMissingValuesWithNoMissingValues() {
        float[] point = { 12.3f, -45.6f };
        int[] missingIndexes = { 1, 1000 }; // second value doesn't matter since numberOfMissingValues is 1o

        double[] result = forest.imputeMissingValues(toDoubleArray(point), 0, missingIndexes);
        assertArrayEquals(new double[] { 0.0, 0.0 }, result);
    }

    @Test
    public void testImputeMissingValuesWithOutputNotReady() {
        double[] point = { 12.3, -45.6 };
        int numberOfMissingValues = 1;
        int[] missingIndexes = { 1, 1000 }; // second value doesn't matter since numberOfMissingValues is 1o

        assertFalse(forest.isOutputReady());
        double[] zero = new double[dimensions];
        assertArrayEquals(zero, forest.imputeMissingValues(point, numberOfMissingValues, missingIndexes));
    }

    @Test
    public void testExtrapolateBasic() {
        doNothing().when(forest).extrapolateBasicCyclic(any(RangeVector.class), anyInt(), anyInt(), anyInt(),
                any(float[].class), any(int[].class), anyDouble());
        doNothing().when(forest).extrapolateBasicSliding(any(RangeVector.class), anyInt(), anyInt(), any(float[].class),
                any(int[].class), anyDouble());

        double[] point = new double[] { 2.0, -3.0 };
        int horizon = 2;
        int blockSize = 1;
        boolean cyclic = true;
        int shingleIndex = 1;

        forest.extrapolateBasic(point, horizon, blockSize, cyclic, shingleIndex);
        verify(forest).extrapolateBasicCyclic(any(RangeVector.class), eq(horizon), eq(blockSize), eq(shingleIndex),
                any(float[].class), any(int[].class), anyDouble());

        forest.extrapolateBasic(point, horizon, blockSize, cyclic);
        verify(forest).extrapolateBasicCyclic(any(RangeVector.class), eq(horizon), eq(blockSize), eq(0),
                any(float[].class), any(int[].class), anyDouble());

        cyclic = false;
        forest.extrapolateBasic(point, horizon, blockSize, cyclic, shingleIndex);
        forest.extrapolateBasic(point, horizon, blockSize, cyclic);
        verify(forest, times(2)).extrapolateBasicSliding(any(RangeVector.class), eq(horizon), eq(blockSize),
                any(float[].class), any(int[].class), anyDouble());
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
                () -> forest.extrapolateBasic((double[]) null, horizon, blockSize, cyclic, shingleIndex));

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
    public void testExtrapolateBasicWithShingleBuilder() {
        doNothing().when(forest).extrapolateBasicCyclic(any(RangeVector.class), anyInt(), anyInt(), anyInt(),
                any(float[].class), any(int[].class), anyDouble());
        doNothing().when(forest).extrapolateBasicSliding(any(RangeVector.class), anyInt(), anyInt(), any(float[].class),
                any(int[].class), anyDouble());

        ShingleBuilder shingleBuilder = new ShingleBuilder(1, 2, true);
        int horizon = 3;

        forest.extrapolateBasic(shingleBuilder, horizon);
        verify(forest, times(1)).extrapolateBasicCyclic(any(RangeVector.class), eq(horizon), eq(1), eq(0),
                any(float[].class), any(int[].class), anyDouble());

        shingleBuilder = new ShingleBuilder(1, 2, false);
        forest.extrapolateBasic(shingleBuilder, horizon);
        verify(forest, times(1)).extrapolateBasicSliding(any(RangeVector.class), eq(horizon), eq(1), any(float[].class),
                any(int[].class), anyDouble());
    }

    @Test
    public void testExtrapolateBasicSliding() {
        int horizon = 3;
        int blockSize = 2;
        RangeVector result = new RangeVector(dimensions * horizon);
        float[] queryPoint = new float[] { 1.0f, -2.0f };
        int[] missingIndexes = new int[blockSize];

        doReturn(new SampleSummary(new float[] { 2.0f, -3.0f }))
                .doReturn(new SampleSummary(new float[] { 4.0f, -5.0f }))
                .doReturn(new SampleSummary(new float[] { 6.0f, -7.0f })).when(forest)
                .getConditionalFieldSummary(aryEq(queryPoint), eq(blockSize), any(int[].class), anyInt(), anyDouble(),
                        any(Boolean.class), any(Boolean.class), anyDouble(), anyInt());

        forest.extrapolateBasicSliding(result, horizon, blockSize, queryPoint, missingIndexes, 1.0);

        float[] expectedResult = new float[] { 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f };
        assertArrayEquals(expectedResult, result.values);
        // test properties of RangeVector as well
        for (int i = 0; i < 6; i++) {
            assert (result.upper[i] >= result.values[i]);
            assert (result.lower[i] <= result.values[i]);
        }
        // validate subsequent operations (typically used in parkservices)
        expectedResult[0] = 0f;
        RangeVector newVector = new RangeVector(expectedResult);
        RangeVector another = new RangeVector(result);
        another.shift(0, -2.0f);
        another.scale(2, 0.25f);
        newVector.scale(2, 0.25f);
        assertArrayEquals(newVector.values, another.values, 1e-6f);
        for (int i = 0; i < 6; i++) {
            assert (another.upper[i] >= another.values[i]);
            assert (another.lower[i] <= another.values[i]);
        }
    }

    @Test
    public void testExtrapolateBasicCyclic() {
        int horizon = 3;
        int blockSize = 2;
        RangeVector result = new RangeVector(dimensions * horizon);
        int shingleIndex = 1;
        float[] queryPoint = new float[] { 1.0f, -2.0f };
        int[] missingIndexes = new int[blockSize];

        doReturn(new SampleSummary(new float[] { 2.0f, -3.0f }))
                .doReturn(new SampleSummary(new float[] { 4.0f, -5.0f }))
                .doReturn(new SampleSummary(new float[] { 6.0f, -7.0f })).when(forest)
                .getConditionalFieldSummary(aryEq(queryPoint), eq(blockSize), any(int[].class), anyInt(), anyDouble(),
                        any(Boolean.class), any(Boolean.class), anyDouble(), anyInt());

        forest.extrapolateBasicCyclic(result, horizon, blockSize, shingleIndex, queryPoint, missingIndexes, 1.0);

        float[] expectedResult = new float[] { -3.0f, 2.0f, -5.0f, 4.0f, -7.0f, 6.0f };
        assertArrayEquals(expectedResult, result.values);
        // test properties of RangeVector as well
        for (int i = 0; i < 6; i++) {
            assert (result.upper[i] >= result.values[i]);
            assert (result.lower[i] <= result.values[i]);
        }
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

        Neighbor neighbor1 = new Neighbor(new float[] { 1, 2 }, 5, indexes1);
        when(((SamplerPlusTree<?, ?>) components.get(0)).getTree().traverse(any(float[].class),
                any(IVisitorFactory.class))).thenReturn(Optional.of(neighbor1));

        Neighbor neighbor2 = new Neighbor(new float[] { 1, 2 }, 5, indexes2);
        when(((SamplerPlusTree<?, ?>) components.get(1)).getTree().traverse(any(float[].class),
                any(IVisitorFactory.class))).thenReturn(Optional.of(neighbor2));

        when(((SamplerPlusTree<?, ?>) components.get(2)).getTree().traverse(any(float[].class),
                any(IVisitorFactory.class))).thenReturn(Optional.empty());

        Neighbor neighbor4 = new Neighbor(new float[] { 2, 3 }, 4, indexes4);
        when(((SamplerPlusTree<?, ?>) components.get(3)).getTree().traverse(any(float[].class),
                any(IVisitorFactory.class))).thenReturn(Optional.of(neighbor4));

        Neighbor neighbor5 = new Neighbor(new float[] { 2, 3 }, 4, indexes5);
        when(((SamplerPlusTree<?, ?>) components.get(4)).getTree().traverse(any(float[].class),
                any(IVisitorFactory.class))).thenReturn(Optional.of(neighbor5));

        for (int i = 5; i < components.size(); i++) {
            when(((SamplerPlusTree<?, ?>) components.get(i)).getTree().traverse(any(float[].class),
                    any(IVisitorFactory.class))).thenReturn(Optional.empty());
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
        verify(forest, times(1)).getNearNeighborsInSample(aryEq(new float[] { 0.1f, 0.2f }),
                eq(Double.POSITIVE_INFINITY));
    }

    @Test
    public void testGetNearNeighborsInSampleInvalid() {
        assertThrows(NullPointerException.class, () -> forest.getNearNeighborsInSample((double[]) null, 101.1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getNearNeighborsInSample(new double[] { 1.1, 2.2 }, -101.1));
        assertThrows(IllegalArgumentException.class,
                () -> forest.getNearNeighborsInSample(new double[] { 1.1, 2.2 }, 0.0));
    }

    @Test
    public void testUpdateOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        RandomCutForest.Builder forestBuilder = RandomCutForest.builder().dimensions(1).numberOfTrees(1).sampleSize(3)
                .timeDecay(0.5).randomSeed(0).parallelExecutionEnabled(false);

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

    @Test
    public void testIsOutputReady() {
        assertFalse(forest.isOutputReady());

        for (int i = 0; i < numberOfTrees / 2; i++) {
            doReturn(true).when(components.get(i)).isOutputReady();
        }
        assertFalse(forest.isOutputReady());

        for (int i = 0; i < numberOfTrees; i++) {
            doReturn(true).when(components.get(i)).isOutputReady();
        }
        assertFalse(forest.isOutputReady());
        when(updateCoordinator.getTotalUpdates()).thenReturn((long) sampleSize);
        assertTrue(forest.isOutputReady());

        // After forest.isOutputReady() returns true once, the result should be cached

        for (int i = 0; i < numberOfTrees; i++) {
            IComponentModel<?, ?> component = components.get(i);
            reset(component);
            doReturn(true).when(component).isOutputReady();
        }
        assertTrue(forest.isOutputReady());
        for (int i = 0; i < numberOfTrees; i++) {
            IComponentModel<?, ?> component = components.get(i);
            verify(component, never()).isOutputReady();
        }
    }

    @Test
    public void testUpdateAfterRoundTrip() {
        int dimensions = 10;
        for (int trials = 0; trials < 10; trials++) {
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).sampleSize(64).build();

            Random r = new Random();
            for (int i = 0; i < new Random(trials).nextInt(3000); i++) {
                forest.update(r.ints(dimensions, 0, 50).asDoubleStream().toArray());
            }

            // serialize + deserialize
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContextEnabled(true);
            mapper.setSaveTreeStateEnabled(true);
            RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));

            // update re-instantiated forest
            for (int i = 0; i < 10000; i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();

                double score = forest.getAnomalyScore(point);
                assertEquals(score, forest2.getAnomalyScore(point), 1e-5);
                forest2.update(point);
                forest.update(point);
            }
        }
    }

    @Test
    public void testUpdateAfterRoundTripWithPause() {
        int dimensions = 10;
        int shingleSize = 5;
        for (int trials = 0; trials < 10; trials++) {
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).sampleSize(64)
                    .shingleSize(shingleSize).internalShinglingEnabled(true).build();
            RandomCutForest reference = RandomCutForest.builder().dimensions(dimensions).sampleSize(64)
                    .shingleSize(shingleSize).internalShinglingEnabled(true).build();

            Random r = new Random();
            for (int i = 0; i < new Random(trials).nextInt(3000); i++) {
                double[] vec = r.ints(dimensions / shingleSize, 0, 50).asDoubleStream().toArray();
                forest.update(vec);
                reference.update(vec);
            }
            assertTrue(forest.isCurrentlySampling());
            forest.pauseSampling();
            assertFalse(forest.isCurrentlySampling());

            // serialize + deserialize
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveExecutorContextEnabled(true);
            mapper.setSaveTreeStateEnabled(true);
            RandomCutForest forest2 = mapper.toModel(mapper.toState(forest));
            assertFalse(forest2.isCurrentlySampling());

            // update re-instantiated forest
            for (int i = 0; i < 1000; i++) {
                double[] point = r.ints(dimensions / shingleSize, 0, 50).asDoubleStream().toArray();

                if (i % 100 == 0) {
                    if (forest2.isCurrentlySampling()) {
                        forest.pauseSampling();
                        forest.resumeSampling();
                    } else {
                        forest.resumeSampling();
                        forest2.resumeSampling();
                    }
                }
                double score = forest.getAnomalyScore(point);
                assertEquals(score, forest2.getAnomalyScore(point), 1e-5);
                forest2.update(point);
                forest.update(point);
                reference.update(point);
            }
            assertArrayEquals(reference.transformToShingledPoint(new float[dimensions / shingleSize]),
                    forest.transformToShingledPoint(new float[dimensions / shingleSize]), 1e-10f);
        }
    }

    @Test
    public void testUpdateAfterRoundTripMediumNodeStore() {
        int dimensions = 5;
        for (int trials = 0; trials < 10; trials++) {
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).numberOfTrees(1).sampleSize(20000)
                    .build();

            Random r = new Random();
            for (int i = 0; i < 30000 + new Random().nextInt(300); i++) {
                forest.update(r.ints(dimensions, 0, 50).asDoubleStream().toArray());
            }

            // serialize + deserialize
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveTreeStateEnabled(true);
            mapper.setSaveExecutorContextEnabled(true);
            RandomCutForestState state = mapper.toState(forest);
            RandomCutForest forest2 = mapper.toModel(state);

            // update re-instantiated forest
            for (int i = 0; i < 10000; i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                double score = forest.getAnomalyScore(point);
                assertEquals(score, forest2.getAnomalyScore(point), 1E-10);
                forest2.update(point);
                forest.update(point);
            }
            List<ConditionalTreeSample> first = forest.getConditionalField(new float[dimensions], 1, new int[1], 1.0);
            List<ConditionalTreeSample> second = forest2.getConditionalField(new float[dimensions], 1, new int[1], 1.0);
            assertEquals(first.size(), second.size());
            for (int i = 0; i < first.size(); i++) {
                assertEquals(first.get(i).pointStoreIndex, second.get(i).pointStoreIndex);
            }
        }
    }

    @Test
    public void testUpdateAfterRoundTripLargeNodeStore() {
        int dimensions = 5;
        for (int trials = 0; trials < 1; trials++) {
            long seed = new Random().nextLong();
            System.out.println(" this seed " + seed);
            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).numberOfTrees(1)
                    .sampleSize(200000).centerOfMassEnabled(true).randomSeed(seed).build();

            Random r = new Random(seed);
            for (int i = 0; i < 300000 + new Random().nextInt(300); i++) {
                forest.update(r.ints(dimensions, 0, 50).asDoubleStream().toArray());
            }

            // serialize + deserialize
            RandomCutForestMapper mapper = new RandomCutForestMapper();
            mapper.setSaveTreeStateEnabled(true);
            mapper.setSaveExecutorContextEnabled(true);
            RandomCutForestState state = mapper.toState(forest);
            RandomCutForest forest2 = mapper.toModel(state);
            assert (forest2.isCenterOfMassEnabled());
            // update re-instantiated forest
            for (int i = 0; i < 10000; i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                double score = forest.getAnomalyScore(point);
                assertEquals(score, forest2.getAnomalyScore(point), 1E-10);
                forest2.update(point);
                forest.update(point);
            }
            List<ICluster<float[]>> first = forest.summarize(10, 1, 5, null);
            System.out.println("DONE 1");
            List<ICluster<float[]>> second = forest2.summarize(10, 1, 1, null);
            assert (abs(first.size() - second.size()) <= 1);
        }
    }

    @Test
    public void testInternalShinglingRotated() {
        RandomCutForest forest = new RandomCutForest.Builder<>().internalShinglingEnabled(true)
                .internalRotationEnabled(true).shingleSize(2).dimensions(4).numberOfTrees(1).build();
        assertThrows(IllegalArgumentException.class, () -> forest.update(new double[] { 0 }));
        forest.update(new double[] { 0.0, -0.0 });
        assertArrayEquals(forest.lastShingledPoint(), new float[] { 0, 0, 0, 0 });
        forest.update(new double[] { 1.0, -1.0 });
        assertArrayEquals(forest.transformIndices(new int[] { 0, 1 }, 2), new int[] { 0, 1 });
        forest.update(new double[] { 2.0, -2.0 });
        assertEquals(forest.nextSequenceIndex(), 3);
        assertArrayEquals(forest.lastShingledPoint(), new float[] { 2, -2, 1, -1 });
        assertArrayEquals(forest.transformToShingledPoint(new float[] { 7, 8 }), new float[] { 2, -2, 7, 8 });
        assertArrayEquals(forest.transformIndices(new int[] { 0, 1 }, 2), new int[] { 2, 3 });
        assertThrows(IllegalArgumentException.class, () -> forest.update(new double[] { 0, 0, 0, 0 }));
    }

    @Test
    public void testComponents() {
        RandomCutForest forest = new RandomCutForest.Builder<>().dimensions(2).sampleSize(10).numberOfTrees(2).build();

        for (IComponentModel model : forest.getComponents()) {
            assertEquals(model.getConfig(Config.BOUNDING_BOX_CACHE_FRACTION), 1.0);
            model.getConfig(Config.TIME_DECAY);
            assertEquals(model.getConfig(Config.TIME_DECAY), 1.0 / 100);
            assertThrows(IllegalArgumentException.class, () -> model.getConfig("foo"));
            assertThrows(IllegalArgumentException.class, () -> model.setConfig("bar", 0));
        }
    }

    @Test
    public void testOutOfOrderUpdate() {
        RandomCutForest forest = new RandomCutForest.Builder<>().dimensions(2).sampleSize(10).numberOfTrees(2).build();
        forest.setTimeDecay(100); // will act almost like a sliding window buffer
        forest.setBoundingBoxCacheFraction(0.2);
        forest.update(new double[] { 20.0, -20.0 }, 20);
        forest.update(new double[] { 0.0, -0.0 }, 0);
        assertEquals(forest.getNearNeighborsInSample(new double[] { 0.0, -0.0 }, 1).size(), 1);
        for (int i = 1; i < 19; i++) {
            forest.update(new double[] { i, -i }, i);
        }
        // the {0,0} point should be flushed out
        assertEquals(forest.getNearNeighborsInSample(new double[] { 0.0, -0.0 }, 1).size(), 0);
        // the {20,-20} point is present still
        assertEquals(forest.getNearNeighborsInSample(new double[] { 20.0, -20.0 }, 1).size(), 1);
    }

    @Test
    public void testFloatingPointRandomCut() {
        int dimensions = 16;
        int numberOfTrees = 41;
        int sampleSize = 64;
        long seed = new Random().nextLong();
        System.out.println(" seed " + seed);
        int dataSize = 4000 * sampleSize;
        double[][] big = generateShingledData(dataSize, dimensions, 2);
        RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).numberOfTrees(numberOfTrees)
                .sampleSize(sampleSize).randomSeed(seed).boundingBoxCacheFraction(1.0).build();

        int num = 0;
        for (double[] point : big) {
            forest.update(point);
        }
    }

    public static double[][] generateShingledData(int size, int dimensions, long seed) {
        double[][] answer = new double[size][];
        int entryIndex = 0;
        boolean filledShingleAtleastOnce = false;
        double[] history = new double[dimensions];
        int count = 0;
        double[] data = getDataD(size + dimensions - 1, 100, 5, seed);
        for (int j = 0; j < size + dimensions - 1; ++j) { // we stream here ....
            history[entryIndex] = data[j];
            entryIndex = (entryIndex + 1) % dimensions;
            if (entryIndex == 0) {
                filledShingleAtleastOnce = true;
            }
            if (filledShingleAtleastOnce) {
                // System.out.println("Adding " + j);
                answer[count++] = getShinglePoint(history, entryIndex, dimensions);
            }
        }
        return answer;
    }

    private static double[] getShinglePoint(double[] recentPointsSeen, int indexOfOldestPoint, int shingleLength) {
        double[] shingledPoint = new double[shingleLength];
        int i = 0;
        for (int j = 0; j < shingleLength; ++j) {
            double point = recentPointsSeen[(j + indexOfOldestPoint) % shingleLength];
            shingledPoint[i++] = point;

        }
        return shingledPoint;
    }

    static double[] getDataD(int num, double amplitude, double noise, long seed) {

        double[] data = new double[num];
        Random noiseprg = new Random(seed);
        for (int i = 0; i < num; i++) {
            data[i] = amplitude * Math.cos(2 * PI * (i + 50) / 1000) + noise * noiseprg.nextDouble();
        }

        return data;
    }
}
