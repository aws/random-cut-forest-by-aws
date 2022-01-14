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

import java.util.BitSet;
import java.util.HashMap;
import java.util.Random;

import static com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.NULL;

public abstract class BoxCache<Point> implements IBoxCache<Point> {

    protected double cacheFraction;
    protected Random random;
    protected long randomSeed;
    protected AbstractBoundingBox<Point>[] cachedBoxes;
    protected HashMap<Integer, Integer> cacheMap;
    protected BitSet bitSet;
    protected int maxSize;

    protected BoxCache(long seed, double cacheFraction, int maxSize) {
        randomSeed = seed;
        this.cacheFraction = cacheFraction;
        this.maxSize = maxSize;
        initialize();
    }

    abstract void initialize();

    public void setCacheFraction(double cacheFraction) {
        this.cacheFraction = cacheFraction;
        initialize();
    }

    boolean isDirectMap() {
        return cacheFraction >= 0.3;
    }

    @Override
    public boolean containsKey(int index) {
        if (cacheFraction == 0.0) {
            return false;
        } else if (cacheFraction == 1.0) {
            return true;
        } else if (isDirectMap()) {
            return !bitSet.get(index);
        } else {
            return cacheMap.containsKey(index);
        }
    }

    @Override
    public void setBox(int index, AbstractBoundingBox<Point> box) {
        if (!containsKey(index)) {
            return;
        }
        if (isDirectMap()) {
            cachedBoxes[index] = box;
        } else {
            cachedBoxes[cacheMap.get(index)] = box;
        }
    }

    @Override
    public AbstractBoundingBox<Point> getBox(int index) {
        if (!containsKey(index)) {
            return null;
        }
        if (isDirectMap()) {
            return cachedBoxes[index];
        } else {
            return cachedBoxes[cacheMap.get(index)];
        }
    }

    abstract void remap(int[] map);

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
        } else {
            if (bitSet != null) {
                BitSet newBitSet = new BitSet(maxSize);
                for (int i = 0; i < map.length; i++) {
                    if (bitSet.get(i)) {
                        if (map[i] != NULL) {
                            newBitSet.set(map[i]);
                        } else {
                            newBitSet.set(i);
                        }
                    }
                }
                bitSet = newBitSet;
            }
            if (cachedBoxes != null) {
                remap(map);
            }
        }
    }

    @Override
    public void addToBox(int index, Point point) {
        if (!containsKey(index)) {
            return;
        }
        if (isDirectMap()) {
            if (cachedBoxes[index] == null) {
                return;
            }
            cachedBoxes[index].addPoint(point);
        } else {
            if (cachedBoxes[cacheMap.get(index)] == null) {
                return;
            }
            cachedBoxes[cacheMap.get(index)].addPoint(point);
        }
    }
}
