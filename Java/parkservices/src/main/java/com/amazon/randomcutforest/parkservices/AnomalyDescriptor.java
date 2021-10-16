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

import com.amazon.randomcutforest.returntypes.DiVector;

@Getter
@Setter
public class AnomalyDescriptor {

    public static int NUMBER_OF_EXPECTED_VALUES = 1;

    // anomalies should have score for postprocessing
    double rcfScore;
    // the following describes the grade of the anomaly in the range [0:1] where
    // 0 is not an anomaly
    double anomalyGrade;

    // same for attribution; this is basic RCF attribution which has high/low
    // information
    DiVector attribution;

    // sequence index (the number of updates to RCF) -- it is possible in imputation
    // that
    // the number of updates more than the input tuples seen by the overall program
    long totalUpdates;

    // timestamp (basically a sequence index, but can be scaled and jittered as in
    // the example);
    // kept as long for potential future use
    long timestamp;

    // if the anomaly is due to timestamp when it is augmented only for current time
    long expectedTimeStamp;

    // confidence, for both anomalies/non-anomalies
    double dataConfidence;

    // number of trees in the forest
    int forestSize;

    // flag indicating if the anomaly is the start of an anomaly or part of a run of
    // anomalies
    boolean startOfAnomaly;

    // flag indicating if the time stamp is in elevated score region to be
    // considered as anomaly
    boolean inHighScoreRegion;

    /**
     * position of the anomaly vis a vis the current time (can be -ve) if anomaly is
     * detected late, which can and should happen sometime; for shingle size 1; this
     * is always 0
     */
    int relativeIndex;

    // a flattened version denoting the basic contribution of each input variable
    // (not shingled) for the
    // time slice indicated by relativeIndex
    double[] relevantAttribution;

    // when time is appended for the anomalous time slice
    double timeAttribution;

    // current values
    double[] currentValues;

    // the values being replaced; may correspond to past
    double[] oldValues;

    // older timestamp if that is replaced
    long oldTimeStamp;

    // expected values, currently set to maximum 1
    double[][] expectedValuesList;

    // likelihood values for the list
    double[] likelihoodOfValues;

    // the threshold used in inference
    double threshold;

    // the below are information used by the RCF these can be useful as explanations
    // of normal points as well
    // internal to RCF, used for passing information in a streaming manner
    double[] expectedRCFPoint;
    double[] rcfPoint;

    public void setCurrentValues(double[] currentValues) {
        this.currentValues = copyIfNotnull(currentValues);
    }

    public void setAttribution(DiVector attribution) {
        this.attribution = new DiVector(attribution);
    }

    public void setOldValues(double[] values) {
        oldValues = copyIfNotnull(values);
    }

    public void setExpectedRCFPoint(double[] point) {
        expectedRCFPoint = copyIfNotnull(point);
    }

    public boolean isExpectedValuesPresent() {
        return expectedValuesList != null;
    }

    public void setRCFPoint(double[] point) {
        rcfPoint = copyIfNotnull(point);
    }

    public void setRelevantAttribution(double[] values) {
        this.relevantAttribution = copyIfNotnull(values);
    }

    public void setExpectedValues(int position, double[] values, double likelihood) {
        checkArgument(position < NUMBER_OF_EXPECTED_VALUES, "Increase size of expected array");
        if (expectedValuesList == null) {
            expectedValuesList = new double[NUMBER_OF_EXPECTED_VALUES][];
        }
        if (likelihoodOfValues == null) {
            likelihoodOfValues = new double[NUMBER_OF_EXPECTED_VALUES];
        }
        expectedValuesList[position] = Arrays.copyOf(values, values.length);
        likelihoodOfValues[position] = likelihood;
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }
}
