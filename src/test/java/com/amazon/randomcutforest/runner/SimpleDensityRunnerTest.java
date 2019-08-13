/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.returntypes.DensityOutput;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SimpleDensityRunnerTest {
    private int numberOfTrees;
    private int sampleSize;
    private int shingleSize;
    private int windowSize;
    private String delimiter;
    private boolean headerRow;
    private SimpleDensityRunner runner;

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
        runner = new SimpleDensityRunner();

        runner.parse(
                "--number-of-trees", Integer.toString(numberOfTrees),
                "--sample-size", Integer.toString(sampleSize),
                "--shingle-size", Integer.toString(shingleSize),
                "--window-size", Integer.toString(windowSize),
                "--delimiter", delimiter,
                "--header-row", Boolean.toString(headerRow)
        );

        in = mock(BufferedReader.class);
        out = mock(PrintWriter.class);
    }
/*
    @Test
    public void testRun() throws IOException {
        when(in.readLine()).thenReturn("a,b")
                .thenReturn("1.0,2.0")
                .thenReturn("4.0,5.0")
                .thenReturn(null);
        runner.run(in, out);
        verify(out).println("a,b,prob_mass_0_up,prob_mass_0_down,prob_mass_1_up,prob_mass_1_down");
        verify(out).println("1.0,2.0,0.0,0.0,0.0,0.0");
        verify(out).println("4.0,5.0,0.0,0.0,0.0,0.0");
    }

    @Test
    public void testWriteHeader() {
        String[] line = new String[] {"a", "b"};
        runner.prepareAlgorithm(2);
        runner.writeHeader(line, out);
        verify(out).println("a,b,prob_mass_0_up,prob_mass_0_down,prob_mass_1_up,prob_mass_1_down");
    }

    @Test
    public void testProcessLine() {
        String[] line = new String[] {"1.0", "2.0"};
        runner.prepareAlgorithm(2);
        runner.processLine(line, out);
        verify(out).println("1.0,2.0,0.0,0.0,0.0,0.0");
    }

    @Test
    public void testSimpleDensityTransformer() {
        RandomCutForest forest = mock(RandomCutForest.class);
        when(forest.getDimensions()).thenReturn(2);

        SimpleDensityRunner.SimpleDensityTransformer transformer = new SimpleDensityRunner.SimpleDensityTransformer(forest);

        DensityOutput expected = new DensityOutput(2, 256);
        expected.probMass.high[0] = 0.0;
        expected.probMass.low[0] = 9.0;
        expected.probMass.high[1] = 8.0;
        expected.probMass.low[1] = 7.0;

        when(forest.getSimpleDensity(1.0, 2.0)).thenReturn(expected);
        assertEquals(Arrays.asList("0.0", "9.0", "8.0", "7.0"), transformer.getResultValues(1.0, 2.0));
        assertEquals(Arrays.asList("prob_mass_0_up", "prob_mass_0_down", "prob_mass_1_up", "prob_mass_1_down"), transformer.getResultColumnNames());
        assertEquals(Arrays.asList("NA", "NA", "NA", "NA"), transformer.getEmptyResultValue());
    }
    */
}
