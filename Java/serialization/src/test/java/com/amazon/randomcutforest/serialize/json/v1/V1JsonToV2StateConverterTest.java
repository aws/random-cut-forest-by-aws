/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.serialize.json.v1;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class V1JsonToV2StateConverterTest {

    private V1JsonToV2StateConverter converter;

    @BeforeEach
    public void setUp() {
        converter = new V1JsonToV2StateConverter();
    }

    @ParameterizedTest
    @EnumSource(V1JsonResource.class)
    public void testConvert(V1JsonResource jsonResource) {
        String resource = jsonResource.getResource();
        try (InputStream is = V1JsonToV2StateConverterTest.class.getResourceAsStream(jsonResource.getResource());
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));) {

            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }

            String json = b.toString();
            RandomCutForestState state = converter.convert(json);

            assertEquals(jsonResource.getDimensions(), state.getDimensions());
            assertEquals(jsonResource.getNumberOfTrees(), state.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), state.getSampleSize());
            RandomCutForest forest = new RandomCutForestMapper().toModel(state,0);

            assertEquals(jsonResource.getDimensions(), forest.getDimensions());
            assertEquals(jsonResource.getNumberOfTrees(), forest.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), forest.getSampleSize());

            // perform a simple validation of the deserialized forest by update and scoring with a few points

            Random random = new Random(0);
            for (int i = 0; i < 10; i++) {
                double[] point = getPoint(jsonResource.getDimensions(),random);
                double score = forest.getAnomalyScore(point);
                assertTrue(score > 0);
                forest.update(point);
            }
        } catch (IOException e) {
            fail("Unable to load JSON resource");
        }
    }

    private double[] getPoint(int dimensions, Random random) {
        double[] point = new double[dimensions];
        for (int i = 0; i < point.length; i++) {
            point[i] = random.nextDouble();
        }
        return point;
    }
}
