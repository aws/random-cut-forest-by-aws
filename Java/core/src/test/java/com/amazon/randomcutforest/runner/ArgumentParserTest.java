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

package com.amazon.randomcutforest.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ArgumentParserTest {

    private ArgumentParser parser;

    @BeforeEach
    public void setUp() {
        parser = new ArgumentParser("runner-class", "runner-description");
    }

    @Test
    public void testNew() {
        assertEquals(100, parser.getNumberOfTrees());
        assertEquals(256, parser.getSampleSize());
        assertEquals(0, parser.getWindowSize());
        assertEquals(0.0, parser.getTimeDecay());
        assertEquals(1, parser.getShingleSize());
        assertFalse(parser.getShingleCyclic());
        assertEquals(",", parser.getDelimiter());
        assertFalse(parser.getHeaderRow());
    }

    @Test
    public void testParse() {
        parser.parse("--number-of-trees", "222", "--sample-size", "123", "--window-size", "50", "--shingle-size", "4",
                "--shingle-cyclic", "true", "--delimiter", "\t", "--header-row", "true");

        assertEquals(222, parser.getNumberOfTrees());
        assertEquals(123, parser.getSampleSize());
        assertEquals(50, parser.getWindowSize());
        assertEquals(0.02, parser.getTimeDecay());
        assertEquals(4, parser.getShingleSize());
        assertTrue(parser.getShingleCyclic());
        assertEquals("\t", parser.getDelimiter());
        assertTrue(parser.getHeaderRow());
    }

    @Test
    public void testParseShortFlags() {
        parser.parse("-n", "222", "-s", "123", "-w", "50", "-g", "4", "-c", "true", "-d", "\t");

        assertEquals(222, parser.getNumberOfTrees());
        assertEquals(123, parser.getSampleSize());
        assertEquals(50, parser.getWindowSize());
        assertEquals(0.02, parser.getTimeDecay());
        assertEquals(4, parser.getShingleSize());
        assertEquals("\t", parser.getDelimiter());
    }
}
