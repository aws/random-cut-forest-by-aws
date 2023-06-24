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

package com.amazon.randomcutforest.util;

import static com.amazon.randomcutforest.util.Weighted.createSample;
import static com.amazon.randomcutforest.util.Weighted.prefixPick;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class WeightedTest {
    private Random rng;
    int size = 10000;
    int heavyIndex;
    ArrayList<Weighted<Integer>> list = new ArrayList<>();

    @BeforeEach
    public void setUp() {
        rng = new Random();
        list = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            list.add(new Weighted<>(i, (float) (0.1 + rng.nextDouble())));
        }
        heavyIndex = size + 7;
        list.add(new Weighted<>(heavyIndex, size));
    }

    @Test
    public void testCreateSample() {
        // forcedSample 0 will return a null list
        assertTrue(createSample(list, 0, 10, 0, 1.0).size() == 0);
        // the following should add the last item first
        List<Weighted<Integer>> sampledList = createSample(list, 0, 10, 0.1, 1.0);
        assertTrue(sampledList.size() > 0);
        assertTrue(sampledList.get(0).index == heavyIndex);
        assertTrue(sampledList.get(0).weight == (float) size);
    }

    @Test
    public void testPrefixPick() {
        double total = list.stream().mapToDouble(e -> e.weight).sum();
        assertTrue(total < 2 * size);

        Weighted<Integer> item = prefixPick(list, size / 3.0);
        assertTrue(item.index < size);
        assertTrue(item.weight <= 1.1);

        // should be the last element
        Weighted<Integer> heavyItem = prefixPick(list, 3.0 * size / 4);
        assertTrue(heavyItem.index == heavyIndex);
        assertTrue(heavyItem.weight == (float) size);

        // checking extreme weights
        heavyItem = prefixPick(list, 2 * size);
        assertTrue(heavyItem.index == heavyIndex);
        assertTrue(heavyItem.weight == (float) size);
    }

}
