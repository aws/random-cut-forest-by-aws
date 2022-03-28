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

import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.fail;

public class V2TRCFToV3StateConverterTest {

    @ParameterizedTest
    @EnumSource(V2TRCFJsonResource.class)
    public void test(V2TRCFJsonResource jsonResource) {

        try (InputStream is = V2TRCFToV3StateConverterTest.class.getResourceAsStream(jsonResource.getResource());
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));) {

            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }

            String json = b.toString();

            ObjectMapper mapper = new ObjectMapper();
            mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
            ThresholdedRandomCutForestState state = mapper.readValue(json, ThresholdedRandomCutForestState.class);

            ThresholdedRandomCutForestMapper mapper1 = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest forest = mapper1.toModel(state);

        } catch (IOException e) {
            fail("Unable to load JSON resource");
        }
    }

}
