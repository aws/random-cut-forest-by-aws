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

package com.amazon.randomcutforest.serialize.json.v1;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Stream;

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
    @MethodSource("args")
    public void testConvert(V1JsonResource jsonResource, Precision precision) {
        String resource = jsonResource.getResource();
        try (InputStream is = V1JsonToV2StateConverterTest.class.getResourceAsStream(jsonResource.getResource());
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));) {

            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }

            String json = b.toString();
            RandomCutForestState state = converter.convert(json, precision);

            assertEquals(jsonResource.getDimensions(), state.getDimensions());
            assertEquals(jsonResource.getNumberOfTrees(), state.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), state.getSampleSize());
            RandomCutForest forest = new RandomCutForestMapper().toModel(state, 0);

            assertEquals(jsonResource.getDimensions(), forest.getDimensions());
            assertEquals(jsonResource.getNumberOfTrees(), forest.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), forest.getSampleSize());

            // perform a simple validation of the deserialized forest by update and scoring
            // with a few points

            Random random = new Random(0);
            for (int i = 0; i < 100; i++) {
                double[] point = getPoint(jsonResource.getDimensions(), random);
                double score = forest.getAnomalyScore(point);
                assertTrue(score > 0);
                forest.update(point);
            }
            String newString = new ObjectMapper().writeValueAsString(new RandomCutForestMapper().toState(forest));
            System.out.println(" Old size " + json.length() + ", new Size " + newString.length()
                    + ", improvement factor " + json.length() / newString.length());
        } catch (IOException e) {
            fail("Unable to load JSON resource");
        }
    }



    @ParameterizedTest
    @MethodSource("args")
    public void testMerge(V1JsonResource jsonResource, Precision precision) {
        String resource = jsonResource.getResource();
        try (InputStream is = V1JsonToV2StateConverterTest.class.getResourceAsStream(jsonResource.getResource());
             BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));) {

            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }

            String json = b.toString();
            int number = new Random().nextInt(10) + 1;
            int testNumberOfTrees = Math.min(100,1 + new Random().nextInt(number*jsonResource.getNumberOfTrees() - 1));
            ArrayList<String> models = new ArrayList<>();

            for(int i=0;i<number; i++){
                models.add(json);
            }

            RandomCutForestState state = converter.convert(models, testNumberOfTrees, precision);

            assertEquals(jsonResource.getDimensions(), state.getDimensions());
            assertEquals(testNumberOfTrees, state.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), state.getSampleSize());
            RandomCutForest forest = new RandomCutForestMapper().toModel(state, 0);

            assertEquals(jsonResource.getDimensions(), forest.getDimensions());
            assertEquals(testNumberOfTrees, forest.getNumberOfTrees());
            assertEquals(jsonResource.getSampleSize(), forest.getSampleSize());

            // perform a simple validation of the deserialized forest by update and scoring
            // with a few points

            Random random = new Random(0);
            for (int i = 0; i < 100; i++) {
                double[] point = getPoint(jsonResource.getDimensions(), random);
                double score = forest.getAnomalyScore(point);
                assertTrue(score > 0);
                forest.update(point);
            }
            int expectedSize = (int) Math.floor(1.0*testNumberOfTrees*json.length()/(number*jsonResource.getNumberOfTrees()));
            String newString = new ObjectMapper().writeValueAsString(new RandomCutForestMapper().toState(forest));
            System.out.println(" Copied " + number + " times, old number of trees " + jsonResource.getNumberOfTrees() + ", new trees " + testNumberOfTrees + ", Expected Old size " + expectedSize + ", new Size " + newString.length());
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

    static Stream<Arguments> args() {
        return jsonParams().flatMap(
                classParameter -> precision().map(testParameter -> Arguments.of(classParameter, testParameter)));
    }

    static Stream<Precision> precision() {
        return Stream.of(Precision.FLOAT_32, Precision.FLOAT_64);
    }

    static Stream<V1JsonResource> jsonParams() {
        return Stream.of(V1JsonResource.FOREST_1, V1JsonResource.FOREST_2);
    }
}
