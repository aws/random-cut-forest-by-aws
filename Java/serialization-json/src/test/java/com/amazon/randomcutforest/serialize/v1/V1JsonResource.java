/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.serialize.v1;

import lombok.Getter;

@Getter
public enum V1JsonResource {

    FOREST_1("forest_1.json", 1, 25, 128), FOREST_2("forest_2.json", 4, 40, 256);

    private final String resource;
    private final int dimensions;
    private final int numberOfTrees;
    private final int sampleSize;

    V1JsonResource(String resource, int dimensions, int numberOfTrees, int sampleSize) {
        this.resource = resource;
        this.dimensions = dimensions;
        this.numberOfTrees = numberOfTrees;
        this.sampleSize = sampleSize;
    }
}
