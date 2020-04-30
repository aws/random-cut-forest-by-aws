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

package com.amazon.randomcutforest.runner;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.PrintWriter;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ImputeRunnerTest {

	private int numberOfTrees;
	private int sampleSize;
	private int windowSize;
	private String delimiter;
	private boolean headerRow;
	private String missingValueMarker;
	private ImputeRunner runner;

	private BufferedReader in;
	private PrintWriter out;

	@BeforeEach
	public void setUp() {
		numberOfTrees = 50;
		sampleSize = 100;
		windowSize = 10;
		delimiter = ",";
		missingValueMarker = "X";
		headerRow = true;
		runner = new ImputeRunner();

		runner.parse("--number-of-trees", Integer.toString(numberOfTrees), "--sample-size",
				Integer.toString(sampleSize), "--window-size", Integer.toString(windowSize), "--delimiter", delimiter,
				"--missing-value-marker", missingValueMarker, "--header-row", Boolean.toString(headerRow));

		in = mock(BufferedReader.class);
		out = mock(PrintWriter.class);
	}

	@Test
	public void testRun() throws Exception {
		when(in.readLine()).thenReturn("a,b").thenReturn("1.0,2.0").thenReturn("4.0,X").thenReturn(null);
		runner.run(in, out);
		verify(out).println("a,b");
		verify(out).println("1.0,2.0");
		verify(out).println("0.0,0.0");
	}

	@Test
	public void testWriteHeader() {
		String[] line = new String[]{"a", "b"};
		runner.prepareAlgorithm(2);
		runner.writeHeader(line, out);
		verify(out).println("a,b");
	}

	@Test
	public void testProcessLine() {
		String[] line = new String[]{"1.0", "2.0"};
		runner.prepareAlgorithm(2);
		runner.processLine(line, out);
		verify(out).println("1.0,2.0");

		line = new String[]{missingValueMarker, "2.0"};
		runner.processLine(line, out);
		verify(out).println("0.0,0.0");
	}

}
