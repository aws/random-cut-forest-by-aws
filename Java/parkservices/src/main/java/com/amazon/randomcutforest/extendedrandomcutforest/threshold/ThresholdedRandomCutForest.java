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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.extendedrandomcutforest.AnomalyDescriptor;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class ThresholdedRandomCutForest{

    protected CorrectorThresholder correctorThresholder;

    protected RandomCutForest forest;

    public ThresholdedRandomCutForest(RandomCutForest.Builder builder, double anomalyRate){
        forest = builder.build();
        checkArgument(!forest.isInternalShinglingEnabled(),"Incorrect setting, not supported");
        correctorThresholder = new CorrectorThresholder(forest,anomalyRate);
        if (forest.getDimensions()/forest.getShingleSize() == 1){
            correctorThresholder.setLowerThreshold(1.1);
        }
    }

    public ThresholdedRandomCutForest(RandomCutForest forest, CorrectorThresholder correctorThresholder){
        this.forest = forest;
        this.correctorThresholder = correctorThresholder;
        correctorThresholder.setForest(forest);
    }

    public AnomalyDescriptor process(double[] point) {
       AnomalyDescriptor result = correctorThresholder.process(point);
       forest.update(point);
       return result;
    }


    public RandomCutForest getForest() {
        return forest;
    }

    public CorrectorThresholder getCorrectorThresholder() {
        return correctorThresholder;
    }

}
