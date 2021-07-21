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

package com.amazon.randomcutforest.examples;

import java.util.Map;
import java.util.TreeMap;

import com.amazon.randomcutforest.examples.dynamicinference.DynamicDensity;
import com.amazon.randomcutforest.examples.dynamicinference.DynamicNearNeighbor;
import com.amazon.randomcutforest.examples.serialization.JsonExample;
import com.amazon.randomcutforest.examples.serialization.ProtostuffExample;

public class Main {

    public static final String ARCHIVE_NAME = "randomcutforest-examples-1.0.jar";

    public static void main(String[] args) throws Exception {
        new Main().run(args);
    }

    private final Map<String, Example> examples;
    private int maxCommandLength;

    public Main() {
        examples = new TreeMap<>();
        maxCommandLength = 0;
        add(new JsonExample());
        add(new ProtostuffExample());
        add(new DynamicDensity());
        add(new DynamicNearNeighbor());
    }

    private void add(Example example) {
        examples.put(example.command(), example);
        if (maxCommandLength < example.command().length()) {
            maxCommandLength = example.command().length();
        }
    }

    public void run(String[] args) throws Exception {
        if (args == null || args.length < 1 || args[0].equals("-h") || args[0].equals("--help")) {
            printUsage();
            return;
        }

        String command = args[0];
        if (!examples.containsKey(command)) {
            throw new IllegalArgumentException("No such example: " + command);
        }

        examples.get(command).run();
    }

    public void printUsage() {
        System.out.printf("Usage: java -cp %s [example]%n", ARCHIVE_NAME);
        System.out.println("Examples:");
        String formatString = String.format("\t %%%ds - %%s%%n", maxCommandLength);
        for (Example example : examples.values()) {
            System.out.printf(formatString, example.command(), example.description());
        }
    }

}
