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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;

import com.amazon.randomcutforest.anomalydetection.DynamicAttributionVisitor;
import com.amazon.randomcutforest.anomalydetection.DynamicScoreVisitor;
import com.amazon.randomcutforest.anomalydetection.SimulatedTransductiveScalarScoreVisitor;
import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDiVectorAccumulator;
import com.amazon.randomcutforest.returntypes.OneSidedConvergingDoubleAccumulator;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.RandomCutTree;

/**
 * DynamicRandomCutForest extends {@link RandomCutForest} to add methods for
 * computing scalar scores using scoring functions which can be passed in as
 * lambdas.
 */
public class DynamicScoringRandomCutForest extends RandomCutForest {

    public static Builder builder() {
        return new Builder();
    }

    protected DynamicScoringRandomCutForest(Builder builder) {
        super(builder);
    }

    /**
     * Create a new DynamicScoringRandomCutForest with optional arguments set to
     * default values.
     *
     * @param dimensions The number of dimension in the input data.
     * @param randomSeed The random seed to use to create the forest random number
     *                   generator
     * @return a new RandomCutForest with optional arguments set to default values.
     */
    public static DynamicScoringRandomCutForest defaultForest(int dimensions, long randomSeed) {
        return builder().dimensions(dimensions).randomSeed(randomSeed).build();
    }

    /**
     * Create a new DynamicScoringRandomCutForest with optional arguments set to
     * default values.
     *
     * @param dimensions The number of dimension in the input data.
     * @return a new RandomCutForest with optional arguments set to default values.
     */
    public static DynamicScoringRandomCutForest defaultForest(int dimensions) {
        return builder().dimensions(dimensions).build();
    }

    /**
     * Score a point using the given scoring functions.
     *
     * @param point                   input point being scored
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    the function that applies if input is equal to
     *                                a previously seen sample in a leaf
     * @param unseen                  if the input does not have a match in the
     *                                leaves
     * @param damp                    damping function based on the duplicity of the
     *                                previously seen samples
     * @return anomaly score
     */
    public double getDynamicScore(double[] point, int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp) {

        checkArgument(ignoreLeafMassThreshold >= 0, "ignoreLeafMassThreshold should be greater than or equal to 0");

        if (!isOutputReady()) {
            return 0.0;
        }

        Function<RandomCutTree, Visitor<Double>> visitorFactory = tree -> new DynamicScoreVisitor(point,
                tree.getRoot().getMass(), ignoreLeafMassThreshold, seen, unseen, damp);
        BinaryOperator<Double> accumulator = Double::sum;

        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Similar to above but now the scoring takes in a function of Bounding Box to
     * probabilities (vector over the dimensions); and produces a score af-if the
     * tree were built using that function (when in reality the tree is an RCF).
     * Changing the defaultRCFgVec function to some other function f() will provide
     * a mechanism of dynamic scoring for trees that are built using f() which is
     * the purpose of TransductiveScalarScore visitor. Note that the answer is an
     * MCMC simulation and is not normalized (because the scoring functions are
     * flexible and unknown) and over a small number of trees the errors can be
     * large specially if vecSep is very far from defaultRCFgVec
     *
     * Given the large number of possible sources of distortion, ignoreLeafThreshold
     * is not supported.
     *
     * @param point  point to be scored
     * @param seen   the score function for seen point
     * @param unseen score function for unseen points
     * @param damp   dampening the score for duplicates
     * @param vecSep the function of (BoundingBox) -&gt; array of probabilities
     * @return the simuated score
     */

    public double getDynamicSimulatedScore(double[] point, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp,
            Function<BoundingBox, double[]> vecSep) {

        if (!isOutputReady()) {
            return 0.0;
        }

        Function<RandomCutTree, Visitor<Double>> visitorFactory = tree -> new SimulatedTransductiveScalarScoreVisitor(
                point, tree.getRoot().getMass(), seen, unseen, damp, CommonUtils::defaultRCFgVecFunction, vecSep);
        BinaryOperator<Double> accumulator = Double::sum;

        Function<Double, Double> finisher = sum -> sum / numberOfTrees;

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Score a point using the given scoring functions. This method will
     * short-circuit before visiting all trees if the scores that are returned from
     * a subset of trees appears to be converging to a given value. See
     * {@link OneSidedConvergingDoubleAccumulator} for more about convergence.
     *
     * @param point                   input point
     * @param precision               controls early convergence
     * @param highIsCritical          this is true for the default scoring function.
     *                                If the user wishes to use a different scoring
     *                                function where anomaly scores are low values
     *                                (for example, height in tree) then this should
     *                                be set to false.
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    scoring function when the input matches some
     *                                tuple in the leaves
     * @param unseen                  scoring function when the input is not found
     * @param damp                    dampening function for duplicates which are
     *                                same as input (applies with seen)
     * @return the dynamic score under sequential early stopping
     */
    public double getApproximateDynamicScore(double[] point, double precision, boolean highIsCritical,
            int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> damp) {

        checkArgument(ignoreLeafMassThreshold >= 0, "ignoreLeafMassThreshold should be greater than or equal to 0");

        if (!isOutputReady()) {
            return 0.0;
        }

        Function<RandomCutTree, Visitor<Double>> visitorFactory = tree -> new DynamicScoreVisitor(point,
                tree.getRoot().getMass(), ignoreLeafMassThreshold, seen, unseen, damp);

        ConvergingAccumulator<Double> accumulator = new OneSidedConvergingDoubleAccumulator(highIsCritical, precision,
                DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<Double, Double> finisher = x -> x / accumulator.getValuesAccepted();

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Same as above, but for dynamic scoring. See the params of
     * getDynamicScoreParallel
     *
     * @param point                   point to be scored
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    score function for seen points
     * @param unseen                  score function for unseen points
     * @param newDamp                 dampening function for duplicates in the seen
     *                                function
     * @return dynamic scoring attribution DiVector
     */
    public DiVector getDynamicAttribution(double[] point, int ignoreLeafMassThreshold,
            BiFunction<Double, Double, Double> seen, BiFunction<Double, Double, Double> unseen,
            BiFunction<Double, Double, Double> newDamp) {

        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        Function<RandomCutTree, Visitor<DiVector>> visitorFactory = tree -> new DynamicAttributionVisitor(point,
                tree.getRoot().getMass(), ignoreLeafMassThreshold, seen, unseen, newDamp);
        BinaryOperator<DiVector> accumulator = DiVector::addToLeft;
        Function<DiVector, DiVector> finisher = x -> x.scale(1.0 / numberOfTrees);

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    /**
     * Atrribution for dynamic sequential scoring; getL1Norm() should agree with
     * getDynamicScoringSequential
     *
     * @param point                   input
     * @param precision               parameter to stop early stopping
     * @param highIsCritical          are high values anomalous (otherwise low
     *                                values are anomalous)
     * @param ignoreLeafEquals        should we ignore leaves with mass equal/below
     *                                threshold
     * @param ignoreLeafMassThreshold said threshold
     * @param seen                    function for scoring points that have been
     *                                seen before
     * @param unseen                  function for scoring points not seen in tree
     * @param newDamp                 dampening function based on duplicates
     * @return attribution DiVector of the score
     */
    public DiVector getApproximateDynamicAttribution(double[] point, double precision, boolean highIsCritical,
            boolean ignoreLeafEquals, int ignoreLeafMassThreshold, BiFunction<Double, Double, Double> seen,
            BiFunction<Double, Double, Double> unseen, BiFunction<Double, Double, Double> newDamp) {

        if (!isOutputReady()) {
            return new DiVector(dimensions);
        }

        Function<RandomCutTree, Visitor<DiVector>> visitorFactory = tree -> new DynamicAttributionVisitor(point,
                tree.getRoot().getMass(), ignoreLeafMassThreshold, seen, unseen, newDamp);

        ConvergingAccumulator<DiVector> accumulator = new OneSidedConvergingDiVectorAccumulator(dimensions,
                highIsCritical, precision, DEFAULT_APPROXIMATE_DYNAMIC_SCORE_MIN_VALUES_ACCEPTED, numberOfTrees);

        Function<DiVector, DiVector> finisher = vector -> vector.scale(1.0 / accumulator.getValuesAccepted());

        return traverseForest(point, visitorFactory, accumulator, finisher);
    }

    public static class Builder extends RandomCutForest.Builder<Builder> {
        public DynamicScoringRandomCutForest build() {
            return new DynamicScoringRandomCutForest(this);
        }
    }

}
