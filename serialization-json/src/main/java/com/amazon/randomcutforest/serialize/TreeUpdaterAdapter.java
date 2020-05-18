/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.serialize;

import java.lang.reflect.Type;
import java.util.Comparator;
import java.util.List;

import com.amazon.randomcutforest.TreeUpdater;
import com.amazon.randomcutforest.sampler.WeightedPoint;
import com.google.gson.Gson;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;

/**
 * Adapter for customizing {@link TreeUpdater} serialization.
 */
public class TreeUpdaterAdapter implements JsonDeserializer<TreeUpdater> {

    @Override
    public TreeUpdater deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {
        TreeUpdater treeUpdater = new Gson().fromJson(json, TreeUpdater.class);
        List<WeightedPoint> points = treeUpdater.getSampler().getWeightedSamples();
        points.sort(Comparator.comparingLong(WeightedPoint::getSequenceIndex));
        points.stream().forEachOrdered(point -> treeUpdater.getTree().addPoint(point));
        return treeUpdater;
    }
}
