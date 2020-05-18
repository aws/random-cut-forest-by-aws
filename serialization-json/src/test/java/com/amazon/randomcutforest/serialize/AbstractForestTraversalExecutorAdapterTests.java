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

package com.amazon.randomcutforest.serialize;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.RandomCutTree;
import com.google.gson.JsonObject;

public class AbstractForestTraversalExecutorAdapterTests {

    private AbstractForestTraversalExecutorAdapter adapter = new AbstractForestTraversalExecutorAdapter();

    @Test
    public void serialize_throw_onUnknownExecutor() {
        assertThrows(RuntimeException.class, () -> {
            adapter.serialize(new AbstractForestTraversalExecutor(null) {

                @Override
                protected void update(double[] pointCopy, long entriesSeen) {
                }

                @Override
                public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                        BinaryOperator<R> accumulator, Function<R, S> finisher) {
                    return null;
                }

                @Override
                public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                        Collector<R, ?, S> collector) {
                    return null;
                }

                @Override
                public <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
                        ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {
                    return null;
                }

                @Override
                public <R, S> S traverseForestMulti(double[] point,
                        Function<RandomCutTree, MultiVisitor<R>> visitorFactory, BinaryOperator<R> accumulator,
                        Function<R, S> finisher) {
                    return null;
                }

                @Override
                public <R, S> S traverseForestMulti(double[] point,
                        Function<RandomCutTree, MultiVisitor<R>> visitorFactory, Collector<R, ?, S> collector) {
                    return null;
                }
            }, String.class, null);
        });
    }

    @Test
    public void deserialize_throw_onUnknownExecutor() {
        JsonObject jsonObject = new JsonObject();
        jsonObject.addProperty(AbstractForestTraversalExecutorAdapter.PROPERTY_EXECUTOR_TYPE, "Unsupported");
        assertThrows(RuntimeException.class, () -> {
            adapter.deserialize(jsonObject, String.class, null);
        });
    }
}
