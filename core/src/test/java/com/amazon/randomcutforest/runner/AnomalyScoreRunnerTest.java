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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.RandomCutForest;

public class AnomalyScoreRunnerTest {

    private int numberOfTrees;
    private int sampleSize;
    private int shingleSize;
    private int windowSize;
    private String delimiter;
    private boolean headerRow;
    private AnomalyScoreRunner runner;

    private BufferedReader in;
    private PrintWriter out;

    @BeforeEach
    public void setUp() {
        numberOfTrees = 50;
        sampleSize = 100;
        shingleSize = 1;
        windowSize = 10;
        delimiter = ",";
        headerRow = true;
        runner = new AnomalyScoreRunner();

        runner.parse("--number-of-trees", Integer.toString(numberOfTrees), "--sample-size",
                Integer.toString(sampleSize), "--shingle-size", Integer.toString(shingleSize), "--window-size",
                Integer.toString(windowSize), "--delimiter", delimiter, "--header-row", Boolean.toString(headerRow));

        in = mock(BufferedReader.class);
        out = mock(PrintWriter.class);
    }

    @Test
    public void testRun() throws IOException {
        when(in.readLine()).thenReturn("a,b,c").thenReturn("1.0,2.0,3.0").thenReturn("4.0,5.0,6.0").thenReturn(null);
        runner.run(in, out);
        verify(out).println("a,b,c,anomaly_score");
        verify(out).println("1.0,2.0,3.0,0.0");
        verify(out).println("4.0,5.0,6.0,0.0");
    }

    @Test
    public void testWriteHeader() {
        String[] line = new String[] { "a", "b", "c" };
        runner.prepareAlgorithm(3);
        runner.writeHeader(line, out);
        verify(out).println("a,b,c,anomaly_score");
    }

    @Test
    public void testProcessLine() {
        String[] line = new String[] { "1.0", "2.0", "3.0" };
        runner.prepareAlgorithm(3);
        runner.processLine(line, out);
        verify(out).println("1.0,2.0,3.0,0.0");
    }

    @Test
    public void testAnomalyScoreTransformer() {
        RandomCutForest forest = mock(RandomCutForest.class);
        AnomalyScoreRunner.AnomalyScoreTransformer transformer = new AnomalyScoreRunner.AnomalyScoreTransformer(forest);

        when(forest.getAnomalyScore(new double[] { 1.0, 2.0, 3.0 })).thenReturn(11.0);
        assertEquals(Collections.singletonList("11.0"), transformer.getResultValues(1.0, 2.0, 3.0));
        assertEquals(Collections.singletonList("anomaly_score"), transformer.getResultColumnNames());
        assertEquals(Collections.singletonList("NA"), transformer.getEmptyResultValue());
    }
}
