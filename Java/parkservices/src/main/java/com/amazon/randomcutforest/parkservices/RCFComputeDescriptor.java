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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.config.ScoringStrategy;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * a basic class that is used to store the internal state of the streaming
 * processing in ThresholdedRandomCutForest and others.
 */
@Getter
@Setter
public class RCFComputeDescriptor extends Point implements IRCFComputeDescriptor {

    ForestMode forestMode = ForestMode.STANDARD;

    TransformMethod transformMethod = TransformMethod.NONE;

    ImputationMethod imputationMethod = ImputationMethod.PREVIOUS;

    ScoringStrategy scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;

    // the most important parameter of the forest
    int shingleSize;

    // the actual dimensions
    int dimension;

    // the inputlength; useful for standalone analysis
    int inputLength;

    // sequence index (the number of updates to RCF) -- it is possible in imputation
    // that
    // the number of updates more than the input tuples seen by the overall program
    long totalUpdates;

    // determines if values can be inputed and or expected point calculated
    boolean reasonableForecast;

    // internal timestamp (basically a sequence index, but can be scaled and
    // jittered as in
    // the example);
    // kept as long for potential future use
    long internalTimeStamp;

    // number of trees in the forest
    int numberOfTrees;

    // current missing values, if any
    int[] missingValues;

    // potential number of imputes before processing current point
    int numberOfNewImputes;

    // actual, potentially transformed point on which compute occurs
    double[] RCFPoint;

    // score for various postprocessing
    double RCFScore;

    // same for attribution; this is basic RCF attribution which has high/low
    // information
    DiVector attribution;

    /**
     * position of the anomaly vis a vis the current time (can be -ve) if anomaly is
     * detected late, which can and should happen sometime; for shingle size 1; this
     * is always 0
     */
    int relativeIndex;

    // useful for detecting noise
    double[] deviations;

    // useful for calibration in RCFCaster
    double[] postDeviations;

    // the multiplication factors to convert RCF representation to actuals/input
    double[] scale;

    // the addition performed (after multiplications) to convert RCF representation
    // to actuals/input
    double[] shift;

    // effects of a specific anomaly
    double[] postShift;

    // how long the effects last
    double transformDecay;

    // expected RCFPoint for the current point
    double[] expectedRCFPoint;

    // internal timestamp of last anomaly
    long lastAnomalyInternalTimestamp;

    // expected point of last anomaly
    double[] lastExpectedRCFPoint;

    // if the anomaly is due to timestamp when it is augmented only for current time
    long expectedTimeStamp;

    // used for streaming imputation
    double[][] imputedPoints;

    public RCFComputeDescriptor(double[] input, long inputTimeStamp) {
        super(input, inputTimeStamp);
    }

    public RCFComputeDescriptor(double[] input, long inputTimeStamp, ForestMode forestMode,
            TransformMethod transformMethod, ImputationMethod imputationMethod) {
        super(input, inputTimeStamp);
        this.forestMode = forestMode;
        this.transformMethod = transformMethod;
        this.imputationMethod = imputationMethod;
    }

    public void setShift(double[] shift) {
        this.shift = copyIfNotnull(shift);
    }

    public void setPostShift(double[] shift) {
        this.postShift = copyIfNotnull(shift);
    }

    public double[] getShift() {
        return copyIfNotnull(shift);
    }

    public void setScale(double[] scale) {
        this.scale = copyIfNotnull(scale);
    }

    public double[] getScale() {
        return copyIfNotnull(scale);
    }

    public double[] getDeltaShift() {
        if (shift == null || postShift == null) {
            return null;
        }
        double[] answer = new double[shift.length];
        for (int i = 0; i < shift.length; i++) {
            answer[i] = postShift[i] - shift[i];
        }
        return answer;
    }

    public double[] getCurrentInput() {
        return copyIfNotnull(currentInput);
    }

    public void setExpectedRCFPoint(double[] point) {
        expectedRCFPoint = copyIfNotnull(point);
    }

    public double[] getExpectedRCFPoint() {
        return copyIfNotnull(expectedRCFPoint);
    }

    public void setRCFPoint(double[] point) {
        RCFPoint = copyIfNotnull(point);
    }

    public double[] getRCFPoint() {
        return copyIfNotnull(RCFPoint);
    }

    public void setLastExpecteRCFdPoint(double[] point) {
        lastExpectedRCFPoint = copyIfNotnull(point);
    }

    public double[] getLastExpectedRCFPoint() {
        return copyIfNotnull(lastExpectedRCFPoint);
    }

    public void setAttribution(DiVector attribution) {
        this.attribution = (attribution == null) ? null : new DiVector(attribution);
    }

    public DiVector getAttribution() {
        return (attribution == null) ? null : new DiVector(attribution);
    }

    public int[] getMissingValues() {
        return (missingValues == null) ? null : Arrays.copyOf(missingValues, missingValues.length);
    }

    public void setMissingValues(int[] values) {
        missingValues = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    public void setImputedPoint(int index, double[] impute) {
        checkArgument(numberOfNewImputes > 0, " no imputation is indicated");
        checkArgument(impute != null && impute.length == inputLength, "incorrect length");
        if (imputedPoints == null) {
            imputedPoints = new double[Math.min(numberOfNewImputes, shingleSize - 1)][];
        }
        checkArgument(imputedPoints.length > index && index >= 0 && imputedPoints[index] == null, "already set!");
        imputedPoints[index] = Arrays.copyOf(impute, inputLength);
    }

    // an explicit copy operation to control the stored state
    public RCFComputeDescriptor copyOf() {
        RCFComputeDescriptor answer = new RCFComputeDescriptor(currentInput, inputTimestamp, forestMode,
                transformMethod, imputationMethod);
        answer.setShingleSize(shingleSize);
        answer.setDimension(dimension);
        answer.setInputLength(inputLength);
        answer.setReasonableForecast(reasonableForecast);
        answer.setAttribution(attribution);
        answer.setRCFPoint(RCFPoint);
        answer.setRCFScore(RCFScore);
        answer.setInternalTimeStamp(internalTimeStamp);
        answer.setExpectedRCFPoint(expectedRCFPoint);
        answer.setNumberOfTrees(numberOfTrees);
        answer.setTotalUpdates(totalUpdates);
        answer.setNumberOfNewImputes(numberOfNewImputes);
        answer.setLastAnomalyInternalTimestamp(lastAnomalyInternalTimestamp);
        answer.setLastExpecteRCFdPoint(lastExpectedRCFPoint);
        answer.setScoringStrategy(scoringStrategy);
        answer.setShift(shift);
        answer.setScale(scale);
        answer.setPostShift(postShift);
        answer.setTransformDecay(transformDecay);
        return answer;
    }
}
