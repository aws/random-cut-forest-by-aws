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

package com.amazon.randomcutforest.state.tree;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.tree.CompactRandomCutTreeFloat;

public class CompactRandomCutTreeFloatMapperTest {

    private static int dimensions = 2;
    private static int capacity = 10;

    private static class TreeProvider implements ArgumentsProvider {
        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext extensionContext) throws Exception {
            IPointStore<float[]> pointStore = PointStore.builder().dimensions(dimensions).capacity(capacity)
                    .shingleSize(1).initialSize(capacity).build();
            List<Integer> indexes = new ArrayList<>();

            for (int i = 0; i < capacity; i++) {
                pointStore.add(new double[] { Math.random(), Math.random() }, 0);
                indexes.add(i);
            }

            Collections.shuffle(indexes);

            List<CompactRandomCutTreeFloat> trees = new ArrayList<>();
            trees.add(new CompactRandomCutTreeFloat.Builder().maxSize(capacity).randomSeed(99L).pointStore(pointStore)
                    .boundingBoxCacheFraction(0.0).centerOfMassEnabled(false).storeSequenceIndexesEnabled(false)
                    .build());
            trees.add(new CompactRandomCutTreeFloat.Builder().maxSize(capacity).randomSeed(99L).pointStore(pointStore)
                    .boundingBoxCacheFraction(0.0).centerOfMassEnabled(false).storeSequenceIndexesEnabled(true)
                    .build());
            trees.add(new CompactRandomCutTreeFloat.Builder().maxSize(capacity).randomSeed(99L).pointStore(pointStore)
                    .boundingBoxCacheFraction(0.0).centerOfMassEnabled(true).storeSequenceIndexesEnabled(false)
                    .build());
            trees.add(new CompactRandomCutTreeFloat.Builder().maxSize(capacity).randomSeed(99L).pointStore(pointStore)
                    .boundingBoxCacheFraction(0.0).centerOfMassEnabled(true).storeSequenceIndexesEnabled(true).build());

            trees.forEach(t -> IntStream.range(0, capacity).forEach(i -> t.addPoint(indexes.get(i), i)));

            CompactRandomCutTreeContext context = new CompactRandomCutTreeContext();
            context.setMaxSize(capacity);
            context.setPointStore(pointStore);
            context.setPrecision(Precision.FLOAT_64);

            return trees.stream().map(t -> Arguments.of(t, context));
        }
    }

    private CompactRandomCutTreeFloatMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new CompactRandomCutTreeFloatMapper();
    }

    @ParameterizedTest
    @ArgumentsSource(TreeProvider.class)
    public void testRoundTrip(CompactRandomCutTreeFloat tree, CompactRandomCutTreeContext context) {
        CompactRandomCutTreeFloat tree2 = mapper.toModel(mapper.toState(tree), context);

        assertEquals(tree.getRoot(), tree2.getRoot());
    }
}
