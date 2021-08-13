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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold.state;

import com.amazon.randomcutforest.extendedrandomcutforest.threshold.BasicThresholder;
import com.amazon.randomcutforest.extendedrandomcutforest.threshold.CorrectorThresholder;
import com.amazon.randomcutforest.state.IStateMapper;
import lombok.Getter;
import lombok.Setter;


@Getter
@Setter
public class CorrectorThresholderMapper implements IStateMapper<CorrectorThresholder, CorrectorThresholderState> {

    @Override
    public CorrectorThresholder toModel(CorrectorThresholderState state, long seed) {
        BasicThresholderMapper mapper = new BasicThresholderMapper();

        BasicThresholder thresholder = mapper.toModel(state.getThresholderState());
        return new CorrectorThresholder(thresholder, state);
    }

    @Override
    public CorrectorThresholderState toState(CorrectorThresholder model) {
        CorrectorThresholderState state = new CorrectorThresholderState();
        BasicThresholderMapper mapper = new BasicThresholderMapper();

        state.setThresholderState(mapper.toState(model.getThresholder()));
        state.setLastAnomalyPoint(model.getLastAnomalyPoint());
        state.setTriggerFactor(model.getTriggerFactor());
        state.setInAnomaly(model.isInAnomaly());
        state.setLastScore(model.getLastScore());
        state.setLastAnomalyTimeStamp(model.getLastAnomalyTimeStamp());
        state.setLastAnomalyScore(model.getLastAnomalyScore());
        state.setLastAnomalyAttribution(model.getLastAnomalyAttribution());
        state.setIgnoreSimilar(model.isIgnoreSimilar());
        state.setIgnoreSimilarFactor(model.getIgnoreSimilarFactor());
        state.setPreviousIsPotentialAnomaly(model.isPreviousIsPotentialAnomaly());
        state.setNumberOfAttributors(model.getNumberOfAttributors());

        return state;
    }

}
