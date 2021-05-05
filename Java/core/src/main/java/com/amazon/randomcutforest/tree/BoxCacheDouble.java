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

package com.amazon.randomcutforest.tree;

import java.util.HashMap;
import java.util.Random;

public class BoxCacheDouble extends BoxCache<double[]> {

    public BoxCacheDouble(long seed, double cacheFraction, int maxSize) {
        super(seed, cacheFraction, maxSize);
        initiateMaps();
    }

    void initiateMaps() {
        cacheRandom = new Random(randomSeed);
        if (cacheFraction < 1.0 && cacheFraction > 0.0) {
            cacheMap = new HashMap<>();
            if (this.cacheFraction >= 0.5) {
                cachedBoxes = new BoundingBox[maxSize];
                int exclude = (int) Math.floor((1.0 - cacheFraction) * maxSize);
                for (int i = 0; i < exclude; i++) {
                    cacheMap.put(cacheRandom.nextInt(maxSize), 0);
                }
            } else {
                int include = (int) Math.ceil(cacheFraction * maxSize);
                cachedBoxes = new BoundingBox[include];
                int count = 0;
                for (int i = 0; i < include; i++) {
                    cacheMap.put(cacheRandom.nextInt(maxSize), count++);
                }
            }
        } else if (cacheFraction == 1.0) {
            cachedBoxes = new BoundingBox[maxSize];
        }
    }

}
