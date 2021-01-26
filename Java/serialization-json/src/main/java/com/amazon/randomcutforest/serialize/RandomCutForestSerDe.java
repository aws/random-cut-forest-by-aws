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

import lombok.Getter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.state.ExecutorContext;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.google.gson.Gson;

/**
 * {@link RandomCutForest} serialization. Internally we use the
 * {@link RandomCutForestMapper} class to convert a RandomCutForest into a
 * corresponding state object, and we use
 * <a href="https://github.com/google/gson">Gson</a> to write the state object
 * as a JSON string. The choice to use Gson is an arbitrary implementation
 * detail, but we expose it so users can customize the Gson output (e.g., by
 * enabling pretty printing).
 */
@Getter
public class RandomCutForestSerDe {

    private RandomCutForestMapper mapper;
    private final Gson gson;

    /**
     * Constructor instantiating objects for default serialization.
     */
    public RandomCutForestSerDe() {
        this(new RandomCutForestMapper(), new Gson());
    }

    /**
     * Create a SerDe instance using the provided mapper and Gson objects.
     * 
     * @param mapper A RandomCutForestMapper instance, used to convert a
     *               RandomCutForest to a corresponding state object.
     * @param gson   A Gson instance that will be used to generate JSON for a given
     *               {@link RandomCutForestState} object.
     */
    public RandomCutForestSerDe(RandomCutForestMapper mapper, Gson gson) {
        this.mapper = mapper;
        this.gson = gson;
    }

    /**
     * Serializes a RCF object to a json string.
     *
     * @param forest A Random Cut Forest
     * @return a json string serialized from the Random Cut Forest.
     */
    public String toJson(RandomCutForest forest) {
        return gson.toJson(mapper.toState(forest));
    }

    /**
     * Deserializes a serialized Random Cut Forest JSON string to a Random Cut
     * Forest object.
     *
     * @param json a json string serialized from a RCF
     * @return a RCF deserialized from the string
     */
    public RandomCutForest fromJson(String json) {
        RandomCutForestState state = gson.fromJson(json, RandomCutForestState.class);
        return mapper.toModel(state);
    }

    /**
     * Deserializes a serialized Random Cut Forest JSON string to a Random Cut
     * Forest object.
     *
     * @param json A json string serialized from a RCF
     * @param seed A random seed value used to initialize the forest.
     * @return a RCF deserialized from the string
     */
    public RandomCutForest fromJson(String json, long seed) {
        RandomCutForestState state = gson.fromJson(json, RandomCutForestState.class);
        return mapper.toModel(state, seed);
    }

    /**
     * Deserializes a serialized Random Cut Forest JSON string to a Random Cut
     * Forest object.
     *
     * @param json    A json string serialized from a RCF
     * @param context An executor context that determines the execution properties
     *                of the deserialized forest.
     * @return a RCF deserialized from the string
     */
    public RandomCutForest fromJson(String json, ExecutorContext context) {
        RandomCutForestState state = gson.fromJson(json, RandomCutForestState.class);
        return mapper.toModel(state, context);
    }

    /**
     * Deserializes a serialized Random Cut Forest JSON string to a Random Cut
     * Forest object.
     *
     * @param json    A json string serialized from a RCF
     * @param context An executor context that determines the execution properties
     *                of the deserialized forest.
     * @param seed    A random seed value used to initialize the forest.
     * @return a RCF deserialized from the string
     */
    public RandomCutForest fromJson(String json, ExecutorContext context, long seed) {
        RandomCutForestState state = gson.fromJson(json, RandomCutForestState.class);
        return mapper.toModel(state, context, seed);
    }
}
