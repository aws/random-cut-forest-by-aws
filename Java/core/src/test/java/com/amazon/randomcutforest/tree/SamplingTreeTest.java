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

package com.amazon.randomcutforest.tree;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.Sequential;
import com.amazon.randomcutforest.sampler.IStreamSampler;

public class SamplingTreeTest {

    private IStreamSampler<double[]> sampler;
    private RandomCutTree tree;
    private SamplingTree<double[]> samplingTree;

    // Create a type alias for mocking
    private interface DoubleArraySampler extends IStreamSampler<double[]> {
    }

    @BeforeEach
    public void setUp() {
        sampler = mock(DoubleArraySampler.class);
        tree = mock(RandomCutTree.class);
        samplingTree = new SamplingTree<>(sampler, tree);
    }

    @Test
    public void testNewUpdaterWithInvalidArgs() {
        assertThrows(NullPointerException.class, () -> new SamplingTree<>(null, tree));
        assertThrows(NullPointerException.class, () -> new SamplingTree<>(sampler, null));
        assertThrows(NullPointerException.class, () -> new SamplingTree<>(null, null));
    }

    @Test
    public void testUpdateRejected() {
        when(sampler.sample(any())).thenReturn(false);
        samplingTree.update(new Sequential<>(new double[] { 4.2, 8.4 }, 1111L));
        verify(tree, never()).addPoint(any(Sequential.class));
        verify(tree, never()).deletePoint(any(Sequential.class));
    }

    @Test
    public void testUpdateWithoutEvictedPoint() {
        double[] point = new double[] { 4.2, 8.4 };
        long sequenceIndex = 1111L;
        when(sampler.sample(any())).thenReturn(true);
        when(sampler.getEvictedPoint()).thenReturn(null);

        Sequential<double[]> seqPoint = new Sequential<>(point, sequenceIndex);
        samplingTree.update(seqPoint);

        verify(tree, times(1)).addPoint(seqPoint);
        verify(tree, never()).deletePoint(any(Sequential.class));
    }

    @Test
    public void testUpdateWithEvictedPoint() {
        double[] point = new double[] { 4.2, 8.4 };
        long sequenceIndex = 1111L;
        Sequential<double[]> seqPoint = new Sequential<>(point, sequenceIndex);
        Sequential<double[]> evictedPoint = new Sequential<>(new double[] { -0.5, 2.222 }, 1110L);

        when(sampler.sample(any())).thenReturn(true);
        when(sampler.getEvictedPoint()).thenReturn(evictedPoint);

        samplingTree.update(seqPoint);

        verify(tree, times(1)).addPoint(seqPoint);
        verify(tree, times(1)).deletePoint(evictedPoint);
    }
}
