/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.df.data;

import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Helper methods that deals with data lists and arrays of values
 */
@Deprecated
public final class DataUtils {
  private DataUtils() { }
  
  /**
   * Computes the sum of the values
   * 计算参数的和
   */
  public static int sum(int[] values) {
    int sum = 0;
    for (int value : values) {
      sum += value;
    }
    
    return sum;
  }
  
  /**
   * foreach i : array1[i] += array2[i]
   * 将array2对应index的值,追加到array1中
   */
  public static void add(int[] array1, int[] array2) {
    Preconditions.checkArgument(array1.length == array2.length, "array1.length != array2.length");
    for (int index = 0; index < array1.length; index++) {
      array1[index] += array2[index];
    }
  }
  
  /**
   * foreach i : array1[i] -= array2[i]
   * 将array2对应index的值,从array1中减去
   */
  public static void dec(int[] array1, int[] array2) {
    Preconditions.checkArgument(array1.length == array2.length, "array1.length != array2.length");
    for (int index = 0; index < array1.length; index++) {
      array1[index] -= array2[index];
    }
  }
  
  /**
   * return the index of the maximum of the array, breaking ties randomly
   * 
   * @param rng
   *          used to break ties
   * @return index of the maximum
   * 找到values中最大值的index位置,当相同的最大值有多个的时候,随机产生一个位置即可
   */
  public static int maxindex(Random rng, int[] values) {
    int max = 0;//最大值
    List<Integer> maxindices = new ArrayList<>();//最大值的index下标.因为可能有多个相同的最大值
    
    for (int index = 0; index < values.length; index++) {
      if (values[index] > max) {
        max = values[index];
        maxindices.clear();
        maxindices.add(index);
      } else if (values[index] == max) {
        maxindices.add(index);
      }
    }

    return maxindices.size() > 1 ? maxindices.get(rng.nextInt(maxindices.size())) : maxindices.get(0);
  }
}
