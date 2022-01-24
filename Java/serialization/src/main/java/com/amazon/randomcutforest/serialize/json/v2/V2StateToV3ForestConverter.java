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

package com.amazon.randomcutforest.serialize.json.v2;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.state.Version.V2_0;
import static com.amazon.randomcutforest.state.Version.V2_1;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;

public class V2StateToV3ForestConverter {

    public RandomCutForest convert(RandomCutForestState v2State) {
        String version = v2State.getVersion();
        checkArgument(version.equals(V2_0) || version.equals(V2_1), "incorrect convertor");
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setCompressionEnabled(v2State.isCompressed());
        return mapper.toModel(v2State);
    }

}
