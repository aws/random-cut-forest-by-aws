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

package com.amazon.randomcutforest.parkservices.state;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.state.IStateMapper;

@Getter
@Setter
public class DiVectorMapper implements IStateMapper<DiVector, DiVectorState> {

    @Override
    public DiVector toModel(DiVectorState state, long seed) {

        if (state.getHigh() == null || state.getLow() == null) {
            return null;
        } else {
            return new DiVector(state.getHigh(), state.getLow());
        }
    }

    @Override
    public DiVectorState toState(DiVector model) {

        DiVectorState state = new DiVectorState();
        if (model != null) {
            state.setHigh(Arrays.copyOf(model.high, model.high.length));
            state.setLow(Arrays.copyOf(model.low, model.low.length));
        }
        return state;
    }

}
