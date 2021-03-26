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

import com.amazon.randomcutforest.sampler.ISampled;
import com.amazon.randomcutforest.sampler.IStreamSampler;
import com.amazon.randomcutforest.tree.ITree;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class SamplerPlusTreeTest {
    @Mock
    private ITree<Integer, double[]> tree;
    @Mock
    private IStreamSampler<Integer> sampler;
    private SamplerPlusTree<Integer, double[]> samplerPlusTree;

    @BeforeEach
    public void setUp() {
        samplerPlusTree = new SamplerPlusTree<>(sampler, tree);
    }

    @Test
    public void testUpdateAddPoint() {
        int pointReference = 2;
        long sequenceIndex = 100L;
        int existingPointReference = 222;
        when(sampler.acceptPoint(sequenceIndex)).thenReturn(true);
        when(sampler.getEvictedPoint()).thenReturn(Optional.empty());
        when(tree.addPoint(pointReference, sequenceIndex)).thenReturn(existingPointReference);

        UpdateResult<Integer> result = samplerPlusTree.update(pointReference, sequenceIndex);
        assertTrue(result.getAddedPoint().isPresent());
        assertEquals(existingPointReference, result.getAddedPoint().get());
        assertFalse(result.getDeletedPoint().isPresent());

        verify(tree, never()).deletePoint(any(), anyLong());
        verify(sampler, times(1)).addPoint(existingPointReference);
    }

    @Test
    public void testUpdateAddAndDeletePoint() {
        int pointReference = 2;
        long sequenceIndex = 100L;
        int existingPointReference = 222;
        int evictedPoint = 333;
        long evictedSequenceIndex = 50L;

        ISampled<Integer> evictedPointSampled = mock(ISampled.class);
        when(evictedPointSampled.getValue()).thenReturn(evictedPoint);
        when(evictedPointSampled.getSequenceIndex()).thenReturn(evictedSequenceIndex);

        when(sampler.acceptPoint(sequenceIndex)).thenReturn(true);
        when(sampler.getEvictedPoint()).thenReturn(Optional.of(evictedPointSampled));
        when(tree.addPoint(pointReference, sequenceIndex)).thenReturn(existingPointReference);
        when(tree.deletePoint(evictedPoint, evictedSequenceIndex)).thenReturn(evictedPoint);

        UpdateResult<Integer> result = samplerPlusTree.update(pointReference, sequenceIndex);
        assertTrue(result.getAddedPoint().isPresent());
        assertEquals(existingPointReference, result.getAddedPoint().get());
        assertTrue(result.getDeletedPoint().isPresent());
        assertEquals(evictedPoint, result.getDeletedPoint().get());

        verify(tree, times(1)).deletePoint(evictedPoint, evictedSequenceIndex);
        verify(sampler, times(1)).addPoint(existingPointReference);
    }

    @Test
    public void testRejectPoint() {
        when(sampler.acceptPoint(anyLong())).thenReturn(false);
        UpdateResult<Integer> result = samplerPlusTree.update(2, 100L);
        assertFalse(result.isStateChange());

        verify(tree, never()).addPoint(any(), anyLong());
        verify(tree, never()).deletePoint(any(), anyLong());
        verify(sampler, never()).addPoint(any());
    }
}
