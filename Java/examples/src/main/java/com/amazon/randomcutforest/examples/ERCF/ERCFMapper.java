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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.threshold.CorrectorThresholder;
import com.amazon.randomcutforest.threshold.state.CorrectorThresholderMapper;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ERCFMapper implements IStateMapper<ExtendedRandomCutForest, ERCFState> {

    @Override
    public ExtendedRandomCutForest toModel(ERCFState state, long seed) {

        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        CorrectorThresholderMapper correctorThresholderMapper = new CorrectorThresholderMapper();

        RandomCutForest forest = randomCutForestMapper.toModel(state.getForestState());
        CorrectorThresholder thresholder = correctorThresholderMapper.toModel(state.thresholderState);
        return  new ExtendedRandomCutForest(forest,thresholder,state.getCount());
    }

    @Override
    public ERCFState toState(ExtendedRandomCutForest model) {
        ERCFState state = new ERCFState();
        RandomCutForestMapper randomCutForestMapper = new RandomCutForestMapper();
        randomCutForestMapper.setPartialTreeStateEnabled(true);
        randomCutForestMapper.setSaveTreeStateEnabled(true);
        randomCutForestMapper.setCompressionEnabled(true);
        randomCutForestMapper.setSaveCoordinatorStateEnabled(true);
        randomCutForestMapper.setSaveExecutorContextEnabled(true);

        state.setForestState(randomCutForestMapper.toState(model.getForest()));

        CorrectorThresholderMapper correctorThresholderMapper =  new CorrectorThresholderMapper();
        state.setThresholderState(correctorThresholderMapper.toState(model.getCorrectorThresholder()));
        state.setCount(model.getCount());
        return state;
    }


}
