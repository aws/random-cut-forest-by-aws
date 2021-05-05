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

public class BoxCache<Point> implements IBoxCache<Point> {

    double cacheFraction;
    Random cacheRandom;
    long randomSeed;
    AbstractBoundingBox<Point>[] cachedBoxes;
    HashMap<Integer, Integer> cacheMap;
    int maxSize;

    protected BoxCache(long seed, double cacheFraction, int maxSize) {
        randomSeed = seed;
        this.cacheFraction = cacheFraction;
        this.maxSize = maxSize;
    }

    @Override
    public boolean inUse(int index) {
        if (cacheFraction == 0.0) {
            return false;
        } else if (cacheFraction == 1.0) {
            return true;
        } else if (cacheFraction >= 0.5) {
            return !cacheMap.containsKey(index);
        } else {
            return cacheMap.containsKey(index);
        }
    }

    @Override
    public void setBox(int index, AbstractBoundingBox<Point> box) {
        if (!inUse(index)) {
            return;
        }
        if (cacheFraction >= 0.5) {
            cachedBoxes[index] = box;
        } else {
            cachedBoxes[cacheMap.get(index)] = box;
        }
    }

    @Override
    public AbstractBoundingBox<Point> getBox(int index) {
        if (!inUse(index)) {
            return null;
        }
        if (cacheFraction >= 0.5) {
            return cachedBoxes[index];
        } else {
            return cachedBoxes[cacheMap.get(index)];
        }
    }

    @Override
    public void swapCaches(int[] map) {
        if (cacheMap != null) {
            HashMap<Integer, Integer> newMap = new HashMap<>();
            for (int i = 0; i < map.length; i++) {
                if (cacheMap.containsKey(i)) {
                    newMap.put(map[i], cacheMap.get(i));
                }
            }
            cacheMap = newMap;
        }
    }

    @Override
    public void addToBox(int index, Point point) {
        if (!inUse(index)) {
            return;
        }
        if (cacheFraction >= 0.5) {
            cachedBoxes[index].addPoint(point);
        } else {
            cachedBoxes[cacheMap.get(index)].addPoint(point);
        }
    }
}
