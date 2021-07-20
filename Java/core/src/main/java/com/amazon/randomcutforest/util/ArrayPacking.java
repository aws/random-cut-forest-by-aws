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

package com.amazon.randomcutforest.util;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static java.lang.Math.min;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class ArrayPacking {

    /**
     * For a given base value, return the smallest int value {@code p} so that
     * {@code base^(p + 1) >= Integer.MAX_VALUE}. If
     * {@code base >= Integer.MAX_VALUE}, return 1.
     * 
     * @param base Compute the approximate log of {@code Integer.MAX_VALUE} in this
     *             base.
     * @return the largest int value {@code p} so that
     *         {@code base^p >= Integer.MAX_VALUE} or 1 if
     *         {@code base >= Integer.MAX_VALUE}.
     */
    public static int logMax(long base) {
        checkArgument(base > 1, "Absolute value of base must be greater than 1");

        int pack = 0;
        long num = base;
        while (num < Integer.MAX_VALUE) {
            num = num * base;
            ++pack;
        }
        return Math.max(pack, 1); // pack can be 0 for max - min being more than Integer.MaxValue
    }

    /**
     * Pack an array of ints. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(int[] inputArray, boolean compress) {
        return pack(inputArray, inputArray.length, compress);
    }

    /**
     * Pack an array of ints. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param length     The length of the output array. Only the first
     *                   {@code length} values in {@code inputArray} will be packed.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(int[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, "inputArray must not be null");
        checkArgument(0 <= length && length <= inputArray.length,
                "length must be between 0 and inputArray.length (inclusive)");

        if (!compress || length < 3) {
            return Arrays.copyOf(inputArray, length);
        }

        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < length; i++) {
            min = min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, length };
        } else {
            int packNum = logMax(base);

            int[] output = new int[3 + (int) Math.ceil(1.0 * length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = length;
            int len = 0;
            int used = 0;
            while (len < length) {
                long code = 0;
                int reach = min(len + packNum - 1, length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = (int) code;
                len += packNum;
            }
            checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }
    }

    /**
     * Pack an array of shorts. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(short[] inputArray, boolean compress) {
        return pack(inputArray, inputArray.length, compress);
    }

    /**
     * Pack an array of shorts. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param length     The length of the output array. Only the first
     *                   {@code length} values in {@code inputArray} will be packed.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(short[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, "inputArray must not be null");
        checkArgument(0 <= length && length <= inputArray.length,
                "length must be between 0 and inputArray.length (inclusive)");

        if (!compress || length < 3) {
            int[] ret = new int[length];
            for (int i = 0; i < length; i++) {
                ret[i] = inputArray[i];
            }
            return ret;
        }

        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < length; i++) {
            min = min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, length };
        } else {
            int packNum = logMax(base);

            int[] output = new int[3 + (int) Math.ceil(1.0 * length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = length;
            int len = 0;
            int used = 0;
            while (len < length) {
                long code = 0;
                int reach = min(len + packNum - 1, length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = (int) code;
                len += packNum;
            }
            checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }
    }

    /**
     * Unpack an array previously created by {@link #pack(int[], int, boolean)}.
     * 
     * @param packedArray An array previously created by
     *                    {@link #pack(int[], int, boolean)}.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked ints.
     */
    public static int[] unpackInts(int[] packedArray, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");

        if (!decompress) {
            return Arrays.copyOf(packedArray, packedArray.length);
        }

        return (packedArray.length < 3) ? unpackInts(packedArray, packedArray.length, decompress)
                : unpackInts(packedArray, packedArray[2], decompress);
    }

    /**
     * Unpack an array previously created by {@link #pack(int[], int, boolean)}.
     * 
     * @param packedArray An array previously created by
     *                    {@link #pack(int[], int, boolean)}.
     * @param length      The desired length of the output array. If this number is
     *                    different from the length of the array that was originally
     *                    packed, then the result will be truncated or padded with
     *                    zeros as needed.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked ints.
     */
    public static int[] unpackInts(int[] packedArray, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkArgument(length >= 0, "incorrect length parameter");

        if (packedArray.length < 3 || !decompress) {
            return Arrays.copyOf(packedArray, length);
        }
        int min = packedArray[0];
        int max = packedArray[1];
        int[] output = new int[length];
        if (min == max) {
            if (packedArray[2] >= length) {
                Arrays.fill(output, min);
            } else {
                for (int i = 0; i < packedArray[2]; i++) {
                    output[i] = min;
                }
            }
        } else {
            long base = ((long) max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < packedArray.length; i++) {
                long code = packedArray[i];
                for (int j = 0; j < packNum && count < min(packedArray[2], length); j++) {
                    output[count++] = (int) (min + code % base);
                    code = (int) (code / base);
                }
            }
        }
        return output;
    }

    private static short[] copyToShort(int[] array, int length) {
        short[] ret = new short[length];
        for (int i = 0; i < Math.min(length, array.length); i++) {
            ret[i] = (short) array[i];
        }
        return ret;
    }

    /**
     * Unpack an array previously created by {@link #pack(short[], int, boolean)}.
     *
     * @param packedArray An array previously created by
     *                    {@link #pack(short[], int, boolean)}.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked shorts.
     */
    public static short[] unpackShorts(int[] packedArray, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");

        if (!decompress) {
            return copyToShort(packedArray, packedArray.length);
        }

        return (packedArray.length < 3) ? unpackShorts(packedArray, packedArray.length, decompress)
                : unpackShorts(packedArray, packedArray[2], decompress);
    }

    /**
     * Unpack an array previously created by {@link #pack(short[], int, boolean)}.
     *
     * @param packedArray An array previously created by
     *                    {@link #pack(short[], int, boolean)}.
     * @param length      The desired length of the output array. If this number is
     *                    different from the length of the array that was originally
     *                    packed, then the result will be truncated or padded with
     *                    zeros as needed.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked ints.
     */
    public static short[] unpackShorts(int[] packedArray, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkArgument(length >= 0, "incorrect length parameter");

        if (packedArray.length < 3 || !decompress) {
            return copyToShort(packedArray, length);
        }
        int min = packedArray[0];
        int max = packedArray[1];
        short[] output = new short[length];
        if (min == max) {
            if (packedArray[2] >= length) {
                Arrays.fill(output, (short) min);
            } else {
                for (int i = 0; i < packedArray[2]; i++) {
                    output[i] = (short) min;
                }
            }
        } else {
            long base = ((long) max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < packedArray.length; i++) {
                long code = packedArray[i];
                for (int j = 0; j < packNum && count < min(packedArray[2], length); j++) {
                    output[count++] = (short) (min + code % base);
                    code = (int) (code / base);
                }
            }
        }
        return output;
    }

    /**
     * Pack an array of doubles into an array of bytes.
     * 
     * @param array An array of doubles.
     * @return An array of bytes representing the original array of doubles.
     */
    public static byte[] pack(double[] array) {
        checkNotNull(array, "array must not be null");
        return pack(array, array.length);
    }

    /**
     * Pack an array of doubles into an array of bytes.
     * 
     * @param array  An array of doubles.
     * @param length The number of doubles in the input array to pack into the
     *               resulting byte array.
     * @return An array of bytes representing the original array of doubles.
     */
    public static byte[] pack(double[] array, int length) {
        checkNotNull(array, "array must not be null");
        checkArgument(0 <= length && length <= array.length,
                "length must be between 0 and inputArray.length (inclusive)");

        ByteBuffer buf = ByteBuffer.allocate(length * Double.BYTES);
        for (int i = 0; i < length; i++) {
            buf.putDouble(array[i]);
        }

        return buf.array();
    }

    /**
     * Pack an array of floats into an array of bytes.
     * 
     * @param array An array of floats.
     * @return An array of bytes representing the original array of floats.
     */
    public static byte[] pack(float[] array) {
        checkNotNull(array, "array must not be null");
        return pack(array, array.length);
    }

    /**
     * Pack an array of floats into an array of bytes.
     * 
     * @param array  An array of floats.
     * @param length The number of doubles in the input array to pack into the
     *               resulting byte array.
     * @return An array of bytes representing the original array of floats.
     */
    public static byte[] pack(float[] array, int length) {
        checkNotNull(array, "array must not be null");
        checkArgument(0 <= length && length <= array.length,
                "length must be between 0 and inputArray.length (inclusive)");

        ByteBuffer buf = ByteBuffer.allocate(length * Float.BYTES);
        for (int i = 0; i < length; i++) {
            buf.putFloat(array[i]);
        }

        return buf.array();
    }

    /**
     * Unpack an array of bytes as an array of doubles.
     * 
     * @param bytes An array of bytes.
     * @return an array of doubles obtained by marshalling consecutive bytes in the
     *         input array into doubles.
     */
    public static double[] unpackDoubles(byte[] bytes) {
        checkNotNull(bytes, "bytes must not be null");
        return unpackDoubles(bytes, bytes.length / Double.BYTES);
    }

    /**
     * Unpack an array of bytes as an array of doubles.
     * 
     * @param bytes  An array of bytes.
     * @param length The desired length of the resulting double array. The input
     *               will be truncated or padded with zeros as needed.
     * @return an array of doubles obtained by marshalling consecutive bytes in the
     *         input array into doubles.
     */
    public static double[] unpackDoubles(byte[] bytes, int length) {
        checkNotNull(bytes, "bytes must not be null");
        checkArgument(length >= 0, "length must be greater than or equal to 0");
        checkArgument(bytes.length % Double.BYTES == 0, "bytes.length must be divisible by Double.BYTES");

        ByteBuffer buf = ByteBuffer.wrap(bytes);
        double[] result = new double[length];
        int m = Math.min(length, bytes.length / Double.BYTES);

        for (int i = 0; i < m; i++) {
            result[i] = buf.getDouble();
        }

        return result;
    }

    /**
     * Unpack an array of bytes as an array of floats.
     * 
     * @param bytes An array of bytes.
     * @return an array of floats obtained by marshalling consecutive bytes in the
     *         input array into floats.
     */
    public static float[] unpackFloats(byte[] bytes) {
        checkNotNull(bytes, "bytes must not be null");
        return unpackFloats(bytes, bytes.length / Float.BYTES);
    }

    /**
     * Unpack an array of bytes as an array of floats.
     * 
     * @param bytes  An array of bytes.
     * @param length The desired length of the resulting float array. The input will
     *               be truncated or padded with zeros as needed.
     * @return an array of doubles obtained by marshalling consecutive bytes in the
     *         input array into floats.
     */
    public static float[] unpackFloats(byte[] bytes, int length) {
        checkNotNull(bytes, "bytes must not be null");
        checkArgument(length >= 0, "length must be greater than or equal to 0");
        checkArgument(bytes.length % Float.BYTES == 0, "bytes.length must be divisible by Float.BYTES");

        ByteBuffer buf = ByteBuffer.wrap(bytes);
        float[] result = new float[length];
        int m = Math.min(length, bytes.length / Float.BYTES);

        for (int i = 0; i < m; i++) {
            result[i] = buf.getFloat();
        }

        return result;
    }
}
