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

import java.lang.reflect.Type;
import java.util.Random;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

/**
 * Adapter for customizing {@link Random} serialization.
 *
 * {@link Random} states are not preserved during serialization and are always
 * re-created, seeded with system time.
 */
public class RandomAdapter implements JsonSerializer<Random>, JsonDeserializer<Random> {

    private Random rng = new Random();

    @Override
    public JsonElement serialize(Random src, Type typeOfSrc, JsonSerializationContext context) {
        return new JsonObject();
    }

    @Override
    public Random deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {
        return new Random(rng.nextLong());
    }
}
