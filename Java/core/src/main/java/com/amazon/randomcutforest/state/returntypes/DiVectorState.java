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

package com.amazon.randomcutforest.state.returntypes;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.anomalydetection.AnomalyAttributionVisitor;

import java.io.Serializable;

/**
 * A DiVector is used when we want to track a quantity in both the positive and
 * negative directions for each dimension in a manifold. For example, when using
 * a {@link AnomalyAttributionVisitor} to compute the attribution of the anomaly
 * score to dimension of the input point, we want to know if the anomaly score
 * attributed to the ith coordinate of the input point is due to that coordinate
 * being unusually high or unusually low.
 *
 * The DiVectorState creates a POJO to be used in serialization.
 */
@Getter
@Setter
public class DiVectorState implements Serializable {
    double[] high;
    double[] low;
}