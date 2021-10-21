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
import com.amazon.randomcutforest.config.TransformMethod;

@Getter
@Setter
public class AnomalyDescriptor extends RCFComputeDescriptor {

    public static int NUMBER_OF_EXPECTED_VALUES = 1;

    // the following describes the grade of the anomaly in the range [0:1] where
    // 0 is not an anomaly
    double anomalyGrade;

    // if the anomaly is due to timestamp when it is augmented only for current time
    long expectedTimeStamp;

    // confidence, for both anomalies/non-anomalies
    double dataConfidence;

    // flag indicating if the anomaly is the start of an anomaly or part of a run of
    // anomalies
    boolean startOfAnomaly;

    // flag indicating if the time stamp is in elevated score region to be
    // considered as anomaly
    boolean inHighScoreRegion;

    // a flattened version denoting the basic contribution of each input variable
    // (not shingled) for the
    // time slice indicated by relativeIndex
    double[] relevantAttribution;

    // when time is appended for the anomalous time slice
    double timeAttribution;

    // the values being replaced; may correspond to past
    double[] pastValues;

    // older timestamp if that is replaced
    long pastTimeStamp;

    // expected values, currently set to maximum 1
    double[][] expectedValuesList;

    // likelihood values for the list
    double[] likelihoodOfValues;

    // the threshold used in inference
    double threshold;

    public AnomalyDescriptor(ForestMode forestMode, TransformMethod transformMethod,
            ImputationMethod imputationMethod) {
        super(forestMode, transformMethod, imputationMethod);
    }

    public AnomalyDescriptor(ForestMode forestMode, TransformMethod transformMethod) {
        this(forestMode, transformMethod, ImputationMethod.PREVIOUS);
    }

    public void setPastValues(double[] values) {
        pastValues = copyIfNotnull(values);
    }

    public boolean isExpectedValuesPresent() {
        return expectedValuesList != null;
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

    public void setDataConfidence(double timeDecay, long valuesSeen, long outputAfter, double dataQuality) {
        long total = valuesSeen;
        double lambda = timeDecay;
        double totalExponent = total * lambda;
        if (totalExponent == 0) {
            dataConfidence = 0.0;
        } else if (totalExponent >= 20) {
            dataConfidence = Math.min(1.0, dataQuality);
        } else {
            double eTotal = Math.exp(totalExponent);
            double confidence = dataQuality * (eTotal - Math.exp(lambda * Math.min(total, outputAfter))) / (eTotal - 1);
            dataConfidence = Math.max(0, confidence);
        }
    }

}
