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

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import com.amazon.randomcutforest.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.ParallelForestTraversalExecutor;
import com.amazon.randomcutforest.SequentialForestTraversalExecutor;

/**
 * Adapter for serializing {@link AbstractForestTraversalExecutor} implementations.
 */
public class AbstractForestTraversalExecutorAdapter implements JsonSerializer<AbstractForestTraversalExecutor>,
    JsonDeserializer<AbstractForestTraversalExecutor> {

    public static final String PROPERTY_EXECUTOR_TYPE = "executor_type";
    public static final String PROPERTY_EXECUTOR = "executor";

    @Override
    public JsonElement serialize(AbstractForestTraversalExecutor src, Type type, JsonSerializationContext context) {
        JsonObject executorJson = new JsonObject();
        if (src instanceof SequentialForestTraversalExecutor) {
            executorJson.addProperty(PROPERTY_EXECUTOR_TYPE, SequentialForestTraversalExecutor.class.getSimpleName());
            executorJson.add(PROPERTY_EXECUTOR, context.serialize(src, SequentialForestTraversalExecutor.class));
        } else if (src instanceof ParallelForestTraversalExecutor) {
            executorJson.addProperty(PROPERTY_EXECUTOR_TYPE, ParallelForestTraversalExecutor.class.getSimpleName());
            executorJson.add(PROPERTY_EXECUTOR, context.serialize(src, ParallelForestTraversalExecutor.class));
        } else {
            throw new IllegalArgumentException("Unsupported executor type " + type.getTypeName());
        }
        return executorJson;
    }

    @Override
    public AbstractForestTraversalExecutor deserialize(JsonElement json, Type type, JsonDeserializationContext ctx) {
        AbstractForestTraversalExecutor executor = null;
        JsonObject executorJson = json.getAsJsonObject();
        String executorType = executorJson.getAsJsonPrimitive(PROPERTY_EXECUTOR_TYPE).getAsString();
        if (SequentialForestTraversalExecutor.class.getSimpleName().equals(executorType)) {
            executor = ctx.deserialize(executorJson.get(PROPERTY_EXECUTOR), SequentialForestTraversalExecutor.class);
        } else if (ParallelForestTraversalExecutor.class.getSimpleName().equals(executorType)) {
            executor = ctx.deserialize(
                executorJson.get(PROPERTY_EXECUTOR), ParallelForestTraversalExecutor.class);
        } else {
            throw new IllegalArgumentException("Unsupported executor type " + type.getTypeName());
        }
        return executor;
    }
}
