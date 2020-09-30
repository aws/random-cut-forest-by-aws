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

package com.amazon.randomcutforest.serialize;

import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.amazon.randomcutforest.ForestState;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.tree.Node;
import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * {@link RandomCutForest} serialization.
 */
public class RandomCutForestSerDe {

    private final Gson gson;

    /**
     * Constructor instantiating objects for default serialization.
     */
    public RandomCutForestSerDe() {
        Set<Class<?>> serializationSkipClasses = Stream.of(BiFunction.class, Node.class, ForkJoinPool.class)
                .collect(Collectors.toSet());
        this.gson = new GsonBuilder().addSerializationExclusionStrategy(new ExclusionStrategy() {
            @Override
            public boolean shouldSkipClass(Class<?> clazz) {
                return serializationSkipClasses.contains(clazz);
            }

            @Override
            public boolean shouldSkipField(FieldAttributes field) {
                return false;
            }
        }).create();
        /*
         * registerTypeAdapter(TreeUpdater.class, new TreeUpdaterAdapter())
         * .registerTypeAdapter(AbstractForestTraversalExecutor.class, new
         * AbstractForestTraversalExecutorAdapter())
         * .registerTypeAdapter(RandomCutForest.class, new RandomCutForestAdapter())
         * .registerTypeAdapter(Random.class, new RandomAdapter()).create();
         * 
         */
    }

    /**
     * Serializes a RCF object to a json string.
     *
     * @param forestState a RCF ForestState object
     * @return a json string serialized from the RCF
     */
    public String toJson(ForestState forestState) {
        return gson.toJson(forestState);
    }

    /**
     * Deserializes a serialized RCF json string to a RCF object.
     *
     * @param json a json string serialized from a RCF
     * @return a RCF deserialized from the string
     */
    public ForestState fromJson(String json) {
        return gson.fromJson(json, ForestState.class);
    }
}
