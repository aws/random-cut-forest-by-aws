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

import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;

@Getter
@Setter
public class AnalysisDescriptor {

    /**
     * the intent of this class is to describe the list of anomalies and the final
     * forecast of some data this is most useful in sequential analysis when that
     * data is processed sequentially
     */
    ArrayList<AnomalyDescriptor> anomalies;
    ForecastDescriptor forecastDescriptor;

    public AnalysisDescriptor(ArrayList<AnomalyDescriptor> anomalies, ForecastDescriptor forecastDescriptor) {
        this.anomalies = anomalies;
        this.forecastDescriptor = forecastDescriptor;
    }

}
