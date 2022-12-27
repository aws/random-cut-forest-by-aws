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

package com.amazon.randomcutforest.parkservices.state;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.protostuff.ProtostuffIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;

public class V2TRCFToV3StateConverterTest {

    private ThresholdedRandomCutForestMapper trcfMapper = new ThresholdedRandomCutForestMapper();

    @ParameterizedTest
    @EnumSource(V2TRCFJsonResource.class)
    public void testJson(V2TRCFJsonResource jsonResource) throws JsonProcessingException {
        String json = getStateFromFile(jsonResource.getResource());
        assertNotNull(json);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        ThresholdedRandomCutForestState state = mapper.readValue(json, ThresholdedRandomCutForestState.class);
        ThresholdedRandomCutForest forest = trcfMapper.toModel(state);
        Random r = new Random(0);
        for (int i = 0; i < 200000; i++) {
            double[] point = r.ints(forest.getForest().getDimensions(), 0, 50).asDoubleStream().toArray();
            forest.process(point, 0L);
        }
        assertNotNull(forest);
    }

    @ParameterizedTest
    @EnumSource(V2TRCFByteBase64Resource.class)
    public void testByteBase64(V2TRCFByteBase64Resource byteBase64Resource) {
        String byteBase64 = getStateFromFile(byteBase64Resource.getResource());
        assertNotNull(byteBase64);
        Schema<ThresholdedRandomCutForestState> trcfSchema = RuntimeSchema
                .getSchema(ThresholdedRandomCutForestState.class);
        byte[] bytes = Base64.getDecoder().decode(byteBase64);
        ThresholdedRandomCutForestState state = trcfSchema.newMessage();
        ProtostuffIOUtil.mergeFrom(bytes, state, trcfSchema);
        ThresholdedRandomCutForest forest = trcfMapper.toModel(state);
        assertNotNull(forest);
    }

    private String getStateFromFile(String resourceFile) {
        try (InputStream is = V2TRCFToV3StateConverterTest.class.getResourceAsStream(resourceFile);
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }
            return b.toString();
        } catch (IOException e) {
            fail("Unable to load resource");
        }
        return null;
    }

}
