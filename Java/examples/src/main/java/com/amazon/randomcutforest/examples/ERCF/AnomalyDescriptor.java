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

package com.amazon.randomcutforest.examples.ERCF;

import com.amazon.randomcutforest.returntypes.DiVector;

public class AnomalyDescriptor {
    // anomalies should have score for postprocessing
    double score;
    // same for attribution; this is basic RCF attribution which has high/low information
    DiVector attribution;
    // timestamp (basically a sequence index); kept as long  for potential future use
    long timeStamp;

    /**
     * position of the anomaly vis a vis the current time (can be -ve) if anomaly is detected late, which can
     * and should happen sometime; for shingle size 1; this is always 0
     * */
    int relativeIndex;

    boolean startOfAnomaly;

    // a flattened version denoting the basic contribution of each input variable (not shingled) for the
    // time slice indicated by relativeIndex
    double[] flattenedAttribution;

    // current values
    double[] currentValues;

    // the values being replaced; may correspond to past
    double[] oldValues;

    double [][] expectedValuesList;
    double[] likelihoodOfValues;
}
