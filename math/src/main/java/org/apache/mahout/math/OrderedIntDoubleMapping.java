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

package org.apache.mahout.math;

import java.io.Serializable;

/**
 * 一个向量肯定由位置和值组成的
 * 比如vector{1,3,5,7,9} 那么5这个就是index=2,value=5
 * 因此对于稀松矩阵,就用两个数组表示非0的数据,并且是按照index从小增大的顺序存储在两个数组中
 * indices存储下标,values存储对应的值
 */
public final class OrderedIntDoubleMapping implements Serializable, Cloneable {

  static final double DEFAULT_VALUE = 0.0;//默认值

  private int[] indices;//存储下标
  private double[] values;//存储下标对应的值
  private int numMappings;//目前两个数组已经用到哪个位置了

  // If true, doesn't allow DEFAULT_VALUEs in the mapping (adding a zero discards it). Otherwise, a DEFAULT_VALUE is
  // treated like any other value.
  //true表示不允许添加默认值到这个数组中
  private boolean noDefault = true;

  OrderedIntDoubleMapping(boolean noDefault) {
    this();
    this.noDefault = noDefault;
  }

  OrderedIntDoubleMapping() {
    // no-arg constructor for deserializer
    this(11);
  }

  OrderedIntDoubleMapping(int capacity) {
    indices = new int[capacity];
    values = new double[capacity];
    numMappings = 0;
  }

  OrderedIntDoubleMapping(int[] indices, double[] values, int numMappings) {
    this.indices = indices;
    this.values = values;
    this.numMappings = numMappings;
  }

  public int[] getIndices() {
    return indices;
  }

  public int indexAt(int offset) {
    return indices[offset];
  }

  public void setIndexAt(int offset, int index) {
    indices[offset] = index;
  }

  public double[] getValues() {
    return values;
  }

  public void setValueAt(int offset, double value) {
    values[offset] = value;
  }


  public int getNumMappings() {
    return numMappings;
  }

  private void growTo(int newCapacity) {
    if (newCapacity > indices.length) {
      int[] newIndices = new int[newCapacity];
      System.arraycopy(indices, 0, newIndices, 0, numMappings);
      indices = newIndices;
      double[] newValues = new double[newCapacity];
      System.arraycopy(values, 0, newValues, 0, numMappings);
      values = newValues;
    }
  }

  //二分法找index在indices中的位置
  private int find(int index) {
    int low = 0;
    int high = numMappings - 1;
    while (low <= high) {
      int mid = low + (high - low >>> 1);
      int midVal = indices[mid];
      if (midVal < index) {
        low = mid + 1;
      } else if (midVal > index) {
        high = mid - 1;
      } else {
        return mid;
      }
    }
    return -(low + 1);
  }

  //找到第index个位置的值
  public double get(int index) {
    int offset = find(index);
    return offset >= 0 ? values[offset] : DEFAULT_VALUE;
  }

  public void set(int index, double value) {
    if (numMappings == 0 || index > indices[numMappings - 1]) {//要插入的index索引位置比 最后一个位置还要大,说明是新增
      if (!noDefault || value != DEFAULT_VALUE) {//说明允许存储默认值,或者该值不是默认值,则进行存储
        if (numMappings >= indices.length) {//是否要扩容数组
          growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
        }
        //设置数组的下标和值
        indices[numMappings] = index;
        values[numMappings] = value;
        ++numMappings;
      }
    } else {//说明是修改
      int offset = find(index);//找到位置
      if (offset >= 0) {//说明有该值已经存过数据了
        insertOrUpdateValueIfPresent(offset, value);
      } else {//说明该值以前没存过数据
        insertValueIfNotDefault(index, offset, value);
      }
    }
  }

  /**
   * Merges the updates in linear time by allocating new arrays and iterating through the existing indices and values
   * and the updates' indices and values at the same time while selecting the minimum index to set at each step.
   * @param updates another list of mappings to be merged in.
   */
  public void merge(OrderedIntDoubleMapping updates) {

    //要合并的参数数组
    int[] updateIndices = updates.getIndices();
    double[] updateValues = updates.getValues();

    int newNumMappings = numMappings + updates.getNumMappings();//最终总数量
    int newCapacity = Math.max((int) (1.2 * newNumMappings), newNumMappings + 1);//扩容

    //获取新的数组
    int[] newIndices = new int[newCapacity];
    double[] newValues = new double[newCapacity];

    int k = 0;//newValues的位置
    int i = 0, j = 0;//i表示目前循环本身到第几个位置了,j表示循环update到哪个位置了
    for (; i < numMappings && j < updates.getNumMappings(); ++k) {
      /**
       * 算法,
       * 一次循环存储一个index和value到新的集合newValues里面
       * 去两个队列中下标最小的
       */
      if (indices[i] < updateIndices[j]) {//说明本身的是最小的,因此存放本身到新队列
        newIndices[k] = indices[i];
        newValues[k] = values[i];
        ++i;//本身下标累加
      } else if (indices[i] > updateIndices[j]) {//说明update的是最小的,因此存放update到新队列
        newIndices[k] = updateIndices[j];
        newValues[k] = updateValues[j];
        ++j;//更新update下标累加
      } else {//说明两个队列的下标相同.则以update队列为准,存储update内容
        newIndices[k] = updateIndices[j];
        newValues[k] = updateValues[j];
        //同时两个队列的下标都累加
        ++i;
        ++j;
      }
    }

    //如果本身队列数据多,则后续的元素都添加到新队列
    for (; i < numMappings; ++i, ++k) {
      newIndices[k] = indices[i];
      newValues[k] = values[i];
    }

    //如果update队列数据多,则后续的元素都添加到新队列
    for (; j < updates.getNumMappings(); ++j, ++k) {
      newIndices[k] = updateIndices[j];
      newValues[k] = updateValues[j];
    }

    //更新最终值
    indices = newIndices;
    values = newValues;
    numMappings = k;
  }

  @Override
  public int hashCode() {
    int result = 0;
    for (int i = 0; i < numMappings; i++) {
      result = 31 * result + indices[i];
      result = 31 * result + (int) Double.doubleToRawLongBits(values[i]);
    }
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof OrderedIntDoubleMapping) {
      OrderedIntDoubleMapping other = (OrderedIntDoubleMapping) o;
      if (numMappings == other.numMappings) {
        for (int i = 0; i < numMappings; i++) {
          if (indices[i] != other.indices[i] || values[i] != other.values[i]) {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(10 * numMappings);
    for (int i = 0; i < numMappings; i++) {
      result.append('(');
      result.append(indices[i]);
      result.append(',');
      result.append(values[i]);
      result.append(')');
    }
    return result.toString();
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public OrderedIntDoubleMapping clone() {
    return new OrderedIntDoubleMapping(indices.clone(), values.clone(), numMappings);
  }

  //追加value的值
  public void increment(int index, double increment) {
    int offset = find(index);
    if (offset >= 0) {
      double newValue = values[offset] + increment;
      insertOrUpdateValueIfPresent(offset, newValue);
    } else {
      insertValueIfNotDefault(index, offset, increment);
    }
  }

  /**
   * 说明该offset位置上一对index和value了
   */
  private void insertValueIfNotDefault(int index, int offset, double value) {
    if (value != DEFAULT_VALUE || !noDefault ) {//如果该值不是默认值,或者该值可以存储默认值
      if (numMappings >= indices.length) {//扩容
        growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
      }
      int at = -offset - 1;
      if (numMappings > at) {
        for (int i = numMappings - 1, j = numMappings; i >= at; i--, j--) {//向后移动,空出at位置
          indices[j] = indices[i];
          values[j] = values[i];
        }
      }
      //at位置存储idnex和value
      indices[at] = index;
      values[at] = value;
      numMappings++;
    }
  }

  //进入该函数,说明offset对应的index和value已经存储值了,因此要更新该值,或者如果该位置变成默认值,则要取消该offset对应的index和value
  private void insertOrUpdateValueIfPresent(int offset, double newValue) {
    if (noDefault && newValue == DEFAULT_VALUE) {//如果新的值是默认值,并且noDefault=true说明不允许保留默认值,则说明没必要存储该值,因此说明以前处理的值没意义了,要取消
      //将offset+1的位置元素,以此移动到offset位置上,即将offset位置的index和value取消了
      for (int i = offset + 1, j = offset; i < numMappings; i++, j++) {
        indices[j] = indices[i];
        values[j] = values[i];
      }
      numMappings--;
    } else {//说明该值有意义或者允许保留默认值,并且是更新该值,则直接更新value即可
      values[offset] = newValue;
    }
  }
}
