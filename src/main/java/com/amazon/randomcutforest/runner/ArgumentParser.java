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

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

/**
 * A utility class for parsing command-line arguments.
 */
public class ArgumentParser {

    public static final String ARCHIVE_NAME = "target/random-cut-forest-1.0.jar";
    private final String runnerClass;
    private final String runnerDescription;
    private final Map<String, Argument<?>> shortFlags;
    private final Map<String, Argument<?>> longFlags;
    private final IntegerArgument numberOfTrees;
    private final IntegerArgument sampleSize;
    private final IntegerArgument windowSize;
    private final IntegerArgument shingleSize;
    private final BooleanArgument shingleCyclic;
    private final StringArgument delimiter;
    private final BooleanArgument headerRow;
    private final IntegerArgument randomSeed;

    /**
     * Create a new ArgumentParser.The runner class and runner description will be used in help text.
     * @param runnerClass       The name of the runner class where this argument parser is being invoked.
     * @param runnerDescription A description of the runner class where this argument parser is being invoked.
     */
    public ArgumentParser(String runnerClass, String runnerDescription) {
        this.runnerClass = runnerClass;
        this.runnerDescription = runnerDescription;
        shortFlags = new HashMap<>();
        longFlags = new HashMap<>();

        numberOfTrees = new IntegerArgument(
            "-n", "--number-of-trees",
            "Number of trees to use in the forest.",
            100,
            n -> checkArgument(n > 0, "number of trees should be greater than 0")
        );

        addArgument(numberOfTrees);

        sampleSize = new IntegerArgument(
            "-s", "--sample-size",
            "Number of points to keep in sample for each tree.",
            256,
            n -> checkArgument(n > 0, "sample size should be greater than 0")
        );

        addArgument(sampleSize);

        windowSize = new IntegerArgument(
            "-w", "--window-size",
            "Window size of the sample or 0 for no window.",
            0,
            n -> checkArgument(n > 0, "window size should be greater than 0")
        );

        addArgument(windowSize);

        shingleSize = new IntegerArgument(
            "-g", "--shingle-size",
            "Shingle size to use.",
            1,
            n -> checkArgument(n > 0, "shingle size should be greater than 0")
        );

        addArgument(shingleSize);

        shingleCyclic = new BooleanArgument(
            "-c", "--shingle-cyclic",
            "Set to 'true' to use cyclic shingles instead of linear shingles.",
            false
        );

        addArgument(shingleCyclic);

        delimiter = new StringArgument(
            "-d", "--delimiter",
            "The character or string used as a field delimiter.",
            ","
        );

        addArgument(delimiter);

        headerRow = new BooleanArgument(
            null, "--header-row",
            "Set to 'true' if the data contains a header row.",
            false
        );

        addArgument(headerRow);

        randomSeed = new IntegerArgument(
            null, "--random-seed",
            "Random seed to use in the Random Cut Forest",
            42
        );

        addArgument(randomSeed);
    }

    /**
     * Add a new argument to this argument parser.
     * @param argument An Argument instance for a command-line argument that should be parsed.
     */
    protected void addArgument(Argument<?> argument) {
        checkNotNull(argument, "argument should not be null");

        checkArgument(argument.getShortFlag() == null || !shortFlags.containsKey(argument.getShortFlag()),
            String.format("An argument mapping already exists for %s", argument.getShortFlag()));

        checkArgument(!longFlags.containsKey(argument.getLongFlag()),
            String.format("An argument mapping already exists for %s", argument.getLongFlag()));

        if (argument.getShortFlag() != null) {
            shortFlags.put(argument.getShortFlag(), argument);
        }

        longFlags.put(argument.getLongFlag(), argument);
    }

    /**
     * Remove the argument with the given long flag from help messages. This allows subclasses to suppress arguments
     * as needed. The argument will still exist in this object with its default value.
     *
     * @param longFlag The long flag corresponding to the argument being removed
     */
    protected void removeArgument(String longFlag) {
        Argument<?> argument = longFlags.get(longFlag);
        if (argument != null) {
            longFlags.remove(longFlag);
            shortFlags.remove(argument.getShortFlag());
        }
    }

    /**
     * Parse the given array of command-line arguments.
     * @param arguments An array of command-line arguments.
     */
    public void parse(String... arguments) {
        int i = 0;
        while (i < arguments.length) {
            String flag = arguments[i];

            try {
                if (shortFlags.containsKey(flag)) {
                    shortFlags.get(flag).parse(arguments[++i]);
                } else if (longFlags.containsKey(flag)) {
                    longFlags.get(flag).parse(arguments[++i]);
                } else if ("-h".equals(flag) || "--help".equals(flag)) {
                    printUsage();
                    Runtime.getRuntime().exit(0);
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + flag);
                }
            } catch (Exception e) {
                printUsageAndExit("%s: %s", e.getClass().getName(), e.getMessage());
            }

            i++;
        }
    }

    /**
     * Print a usage message to STDOUT.
     */
    public void printUsage() {
        System.out.println(String.format("Usage: java -cp %s %s [options] < input_file > output_file", ARCHIVE_NAME,
            runnerClass));
        System.out.println();
        System.out.println(runnerDescription);
        System.out.println();
        System.out.println("Options:");

        longFlags.values().stream()
            .map(Argument::getHelpMessage)
            .sorted()
            .forEach(msg -> System.out.println("\t" + msg));

        System.out.println();
        System.out.println("\t--help, -h: Print this help message and exit.");
    }

    /**
     * Print an error message, the usage message, and exit the application.
     * @param errorMessage  An error message to show the user.
     * @param formatObjects An array of format objects that will be interpolated into the error message using
     *                      {@link String#format}.
     */
    public void printUsageAndExit(String errorMessage, Object... formatObjects) {
        System.err.println("Error: " + String.format(errorMessage, formatObjects));
        printUsage();
        System.exit(1);
    }

    /**
     * @return the user-specified value of the number-of-trees parameter.
     */
    public int getNumberOfTrees() {
        return numberOfTrees.getValue();
    }

    /**
     * @return the user-specified value of the sample-size parameter.
     */
    public int getSampleSize() {
        return sampleSize.getValue();
    }

    /**
     * @return the user-specified value of the window-size parameter
     */
    public int getWindowSize() {
        return windowSize.getValue();
    }

    /**
     * @return the user-specified value of the lambda parameter
     */
    public double getLambda() {
        if (getWindowSize() > 0) {
            return 1.0 / getWindowSize();
        } else {
            return 0.0;
        }
    }

    /**
     * @return the user-specified value of the shingle-size parameter
     */
    public int getShingleSize() {
        return shingleSize.getValue();
    }

    /**
     * @return the user-specified value of the shingle-cyclic parameter
     */
    public boolean getShingleCyclic() {
        return shingleCyclic.getValue();
    }

    /**
     * @return the user-specified value of the delimiter parameter
     */
    public String getDelimiter() {
        return delimiter.getValue();
    }

    /**
     * @return the user-specified value of the header-row parameter
     */
    public boolean getHeaderRow() {
        return headerRow.getValue();
    }

    /**
     * @return the user-specified value of the random-seed parameter
     */
    public int getRandomSeed() {
        return randomSeed.getValue();
    }

    public static class Argument<T> {

        private final String shortFlag;
        private final String longFlag;
        private final String description;
        private final T defaultValue;
        private final Function<String, T> parseFunction;
        private final Consumer<T> validateFunction;
        private T value;

        public Argument(String shortFlag, String longFlag, String description, T defaultValue,
                        Function<String, T> parseFunction, Consumer<T> validateFunction) {
            this.shortFlag = shortFlag;
            this.longFlag = longFlag;
            this.description = description;
            this.defaultValue = defaultValue;
            this.parseFunction = parseFunction;
            this.validateFunction = validateFunction;
            value = defaultValue;
        }

        public Argument(String shortFlag, String longFlag, String description, T defaultValue,
                        Function<String, T> parseFunction) {
            this(shortFlag, longFlag, description, defaultValue, parseFunction, t -> {
            });
        }

        public String getShortFlag() {
            return shortFlag;
        }

        public String getLongFlag() {
            return longFlag;
        }

        public String getDescription() {
            return description;
        }

        public T getDefaultValue() {
            return defaultValue;
        }

        public String getHelpMessage() {
            if (shortFlag != null) {
                return String.format("%s, %s: %s (default: %s)", longFlag, shortFlag, description, defaultValue);
            } else {
                return String.format("%s: %s (default: %s)", longFlag, description, defaultValue);
            }
        }

        public void parse(String string) {
            value = parseFunction.apply(string);
            validateFunction.accept(value);
        }

        public T getValue() {
            return value;
        }
    }

    public static class StringArgument extends Argument<String> {
        public StringArgument(String shortFlag, String longFlag, String description, String defaultValue,
                              Consumer<String> validateFunction) {
            super(shortFlag, longFlag, description, defaultValue, x -> x, validateFunction);
        }

        public StringArgument(String shortFlag, String longFlag, String description, String defaultValue) {
            super(shortFlag, longFlag, description, defaultValue, x -> x);
        }
    }

    public static class BooleanArgument extends Argument<Boolean> {
        public BooleanArgument(String shortFlag, String longFlag, String description, boolean defaultValue) {
            super(shortFlag, longFlag, description, defaultValue, Boolean::parseBoolean);
        }
    }

    public static class IntegerArgument extends Argument<Integer> {
        public IntegerArgument(String shortFlag, String longFlag, String description, int defaultValue,
                               Consumer<Integer> validateFunction) {
            super(shortFlag, longFlag, description, defaultValue, Integer::parseInt, validateFunction);
        }

        public IntegerArgument(String shortFlag, String longFlag, String description, int defaultValue) {
            super(shortFlag, longFlag, description, defaultValue, Integer::parseInt);
        }
    }

    public static class DoubleArgument extends Argument<Double> {
        public DoubleArgument(String shortFlag, String longFlag, String description, double defaultValue,
                              Consumer<Double> validateFunction) {
            super(shortFlag, longFlag, description, defaultValue, Double::parseDouble, validateFunction);
        }

        public DoubleArgument(String shortFlag, String longFlag, String description, double defaultValue) {
            super(shortFlag, longFlag, description, defaultValue, Double::parseDouble);
        }
    }
}
