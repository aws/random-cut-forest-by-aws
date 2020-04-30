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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import software.amazon.ion.system.IonBinaryWriterBuilder;
import software.amazon.ion.system.IonReaderBuilder;
import software.amazon.ion.system.IonTextWriterBuilder;
import software.amazon.ion.IonException;
import software.amazon.ion.IonReader;
import software.amazon.ion.IonWriter;
import com.amazon.randomcutforest.AbstractForestTraversalExecutor;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.TreeUpdater;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.ion.IonObjectMapper;
import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import com.amazon.randomcutforest.tree.Node;

/**
 * {@link RandomCutForest} serialization.
 */
public class RandomCutForestSerDe {

    private final Gson gson;

    /**
     * Constructor instantiating objects for default serialization.
     */
    public RandomCutForestSerDe() {
        Set<Class<?>> serializationSkipClasses = Stream
            .of(BiFunction.class, Node.class, ForkJoinPool.class)
            .collect(Collectors.toSet());
        this.gson = new GsonBuilder()
            .addSerializationExclusionStrategy(
                new ExclusionStrategy() {
                    @Override
                    public boolean shouldSkipClass(Class<?> clazz) {
                        return serializationSkipClasses.contains(clazz);
                    }

                    @Override
                    public boolean shouldSkipField(FieldAttributes field) {
                        return false;
                    }
                }
            )
            .registerTypeAdapter(TreeUpdater.class, new TreeUpdaterAdapter())
            .registerTypeAdapter(AbstractForestTraversalExecutor.class, new AbstractForestTraversalExecutorAdapter())
            .registerTypeAdapter(RandomCutForest.class, new RandomCutForestAdapter())
            .registerTypeAdapter(Random.class, new RandomAdapter())
            .create();
    }

    /**
     * Serializes a RCF object to a json string.
     *
     * @param rcf a RCF object
     * @return a json string serialized from the RCF
     */
    public String toJson(RandomCutForest rcf) {
        return gson.toJson(rcf);
    }

    /**
     * Serializes a RCF object to ion binary.
     *
     * @param rcf a RCF object
     * @return ion binary serialized from the RCF
     * @throws IonException when the input cannot be serialized
     */
    public byte[] toIon(RandomCutForest rcf) {
        try {
            IonReaderBuilder readerBuilder = IonReaderBuilder.standard();
            IonReader reader = readerBuilder.build(toJson(rcf));
            IonObjectMapper mapper = new IonObjectMapper();
            mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
            SerializedRandomCutForest srcf = mapper.readValue(reader, SerializedRandomCutForest.class);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            IonWriter ionWriter = IonBinaryWriterBuilder.standard().build(out);
            mapper.writeValue(ionWriter, srcf);
            return out.toByteArray();
        } catch (IOException e) {
            throw new IonException(e);
        }
    }

    /**
     * Deserializes a serialized RCF json string to a RCF object.
     *
     * @param json a json string serialized from a RCF
     * @return a RCF deserialized from the string
     */
    public RandomCutForest fromJson(String json) {
        return gson.fromJson(json, RandomCutForest.class);
    }

    /**
     * Deserializes a serialized RCF ion binary to a RCF object.
     *
     * @param ionBinary ion binary serialized from a RCF
     * @return a RCF deserialized from the byte array
     * @throws IonException when the input cannot be deserialized
     */
    public RandomCutForest fromIon(byte[] ionBinary) {
        try {
            IonReaderBuilder readerBuilder = IonReaderBuilder.standard();
            IonReader reader = readerBuilder.build(ionBinary);
            IonObjectMapper mapper = new IonObjectMapper();
            mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
            SerializedRandomCutForest srcf = mapper.readValue(reader, SerializedRandomCutForest.class);
            StringBuilder out = new StringBuilder();
            IonWriter ionWriter = IonTextWriterBuilder.json().build(out);
            mapper.writeValue(ionWriter, srcf);
            return gson.fromJson(out.toString(), RandomCutForest.class);
        } catch (IOException e) {
            throw new IonException(e);
        }
    }
}
