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

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class GenericAnomalyDescriptor<P> {

    P representative;

    double score;

    double threshold;

    double anomalyGrade;

    public GenericAnomalyDescriptor(P representative, double score, double threshold, double anomalyGrade) {
        this.representative = representative;
        this.score = score;
        this.threshold = threshold;
        this.anomalyGrade = anomalyGrade;
    }

}
