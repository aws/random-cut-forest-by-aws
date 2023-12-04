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

package com.amazon.randomcutforest.parkservices.returntypes;

import java.util.List;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.util.Weighted;

@Getter
@Setter
public class GenericAnomalyDescriptor<P> {

    // the following corresponds to the list of extected points in AnomalyDetector,
    // which is returned from
    // TRCF. The list corresponds to plausible values (cluster centers) and a weight
    // representing the likelihood
    // The list is sorted in decreasing order of likelihood. Most often, the first
    // element should suffice.
    // in case of an anomalous point, however the information here can provide more
    // insight
    List<Weighted<P>> representativeList;

    // standard, as in AnomalyDetector; we do not recommend attempting to
    // disambiguate scores of non-anomalous
    // points. Note that scores can be low.
    double score;

    // standard as in AnomalyDetector
    double threshold;

    // a value between [0,1] indicating the strength of the anomaly, it can be
    // viewed as a confidence score
    // projected by the algorithm.
    double anomalyGrade;

    public GenericAnomalyDescriptor(List<Weighted<P>> representative, double score, double threshold,
            double anomalyGrade) {
        this.representativeList = representative;
        this.score = score;
        this.threshold = threshold;
        this.anomalyGrade = anomalyGrade;
    }

}
