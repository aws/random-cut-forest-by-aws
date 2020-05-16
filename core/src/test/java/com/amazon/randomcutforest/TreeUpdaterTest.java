/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.sampler.SimpleStreamSampler;
import com.amazon.randomcutforest.sampler.WeightedPoint;
import com.amazon.randomcutforest.tree.RandomCutTree;

public class TreeUpdaterTest {

    private SimpleStreamSampler sampler;
    private RandomCutTree tree;
    private int randomSeed;
    private TreeUpdater updater;

    @BeforeEach
    public void setUp() {
        sampler = mock(SimpleStreamSampler.class);
        tree = mock(RandomCutTree.class);
        updater = new TreeUpdater(sampler, tree);
    }

    @Test
    public void testNewUpdaterWithInvalidArgs() {
        assertThrows(NullPointerException.class, () -> new TreeUpdater(null, tree));
        assertThrows(NullPointerException.class, () -> new TreeUpdater(sampler, null));
        assertThrows(NullPointerException.class, () -> new TreeUpdater(null, null));
    }

    @Test
    public void testUpdateRejected() {
        when(sampler.sample(any(), anyInt())).thenReturn(null);
        updater.update(new double[] { 4.2, 8.4 }, 1111L);
        verify(tree, never()).addPoint(any());
        verify(tree, never()).deletePoint(any());
    }

    @Test
    public void testUpdateRejectedWithoutEvictedPoint() {
        double[] point = new double[] { 4.2, 8.4 };
        long sequenceIndex = 1111L;
        WeightedPoint sampledPoint = new WeightedPoint(point, sequenceIndex, 0.001);
        when(sampler.sample(any(), anyInt())).thenReturn(sampledPoint);
        when(sampler.getEvictedPoint()).thenReturn(null);

        updater.update(point, sequenceIndex);

        verify(tree, times(1)).addPoint(sampledPoint);
        verify(tree, never()).deletePoint(any());
    }

    @Test
    public void testUpdateRejectedWithEvictedPoint() {
        double[] point = new double[] { 4.2, 8.4 };
        long sequenceIndex = 1111L;
        WeightedPoint sampledPoint = new WeightedPoint(point, sequenceIndex, 0.001);
        WeightedPoint evictedPoint = new WeightedPoint(new double[] { -0.5, 2.222 }, 1110L, 0.123);

        when(sampler.sample(any(), anyInt())).thenReturn(sampledPoint);
        when(sampler.getEvictedPoint()).thenReturn(evictedPoint);

        updater.update(point, sequenceIndex);

        verify(tree, times(1)).addPoint(sampledPoint);
        verify(tree, times(1)).deletePoint(evictedPoint);
    }
}
