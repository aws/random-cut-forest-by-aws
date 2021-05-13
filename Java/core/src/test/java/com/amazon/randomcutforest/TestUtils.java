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


import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.INodeView;
import com.amazon.randomcutforest.tree.RandomCutTree;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collector;

public class TestUtils {
    public static final double EPSILON = 1e-6;

    /** Return a visitor that does nothing. */
    public static final Function<RandomCutTree, Visitor<Double>> DUMMY_VISITOR_FACTORY =
            tree ->
                    new Visitor<Double>() {
                        @Override
                        public void accept(INodeView node, int depthOfNode) {}

                        @Override
                        public Double getResult() {
                            return Double.NaN;
                        }
                    };

    /** Return a visitor that does nothing. */
    public static final VisitorFactory<Double> DUMMY_GENERIC_VISITOR_FACTORY =
            tree ->
                    new Visitor<Double>() {
                        @Override
                        public void accept(INodeView node, int depthOfNode) {}

                        @Override
                        public Double getResult() {
                            return Double.NaN;
                        }
                    };

    /** Return a multi-visitor that does nothing. */
    public static final Function<RandomCutTree, MultiVisitor<Double>> DUMMY_MULTI_VISITOR_FACTORY =
            tree ->
                    new MultiVisitor<Double>() {
                        @Override
                        public void accept(INodeView node, int depthOfNode) {}

                        @Override
                        public Double getResult() {
                            return Double.NaN;
                        }

                        @Override
                        public boolean trigger(INodeView node) {
                            return false;
                        }

                        @Override
                        public MultiVisitor<Double> newCopy() {
                            return null;
                        }

                        @Override
                        public void combine(MultiVisitor<Double> other) {}
                    };

    /** A collector that accumulates values into a sorted list. */
    public static final Collector<Double, List<Double>, List<Double>> SORTED_LIST_COLLECTOR =
            Collector.of(
                    ArrayList::new,
                    List::add,
                    (left, right) -> {
                        left.addAll(right);
                        return left;
                    },
                    list -> {
                        list.sort(Double::compare);
                        return list;
                    });

    /**
     * Return a converging accumulator that converges after seeing numberOfEntries values. The
     * returned value is the sum of all accepted values.
     *
     * @param numberOfEntries The number of entries that need to be accepted for this accumulator to
     *     converge.
     * @return a new converging accumulator that converges after seeing numberOfEntries values.
     */
    public static ConvergingAccumulator<Double> convergeAfter(int numberOfEntries) {
        return new ConvergingAccumulator<Double>() {
            private int valuesAccepted = 0;
            private double total = 0.0;

            @Override
            public void accept(Double value) {
                valuesAccepted++;
                total += value;
            }

            @Override
            public boolean isConverged() {
                return valuesAccepted >= numberOfEntries;
            }

            @Override
            public int getValuesAccepted() {
                return valuesAccepted;
            }

            @Override
            public Double getAccumulatedValue() {
                return total;
            }
        };
    }

    /** Return a multi-visitor that does nothing. */
    public static final MultiVisitorFactory<Double> DUMMY_GENERIC_MULTI_VISITOR_FACTORY =
            tree ->
                    new MultiVisitor<Double>() {
                        @Override
                        public void accept(INodeView node, int depthOfNode) {}

                        @Override
                        public Double getResult() {
                            return Double.NaN;
                        }

                        @Override
                        public boolean trigger(INodeView node) {
                            return false;
                        }

                        @Override
                        public MultiVisitor<Double> newCopy() {
                            return null;
                        }

                        @Override
                        public void combine(MultiVisitor<Double> other) {}
                    };
}
