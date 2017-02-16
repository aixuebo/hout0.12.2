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

package org.apache.mahout.cf.taste.impl.common;

import java.io.Serializable;

/**
 * <p>
 * A simple class that can keep track of a running average of a series of numbers. One can add to or remove
 * from the series, as well as update a datum in the series. The class does not actually keep track of the
 * series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 * </p>
 * 计算数量和平均值
 */
public class FullRunningAverage implements RunningAverage, Serializable {
  
  private int count;
  private double average;
  
  public FullRunningAverage() {
    this(0, Double.NaN);
  }

  public FullRunningAverage(int count, double average) {
    this.count = count;
    this.average = average;
  }

  /**
   * @param datum
   *          new item to add to the running average
   */
  @Override
  public synchronized void addDatum(double datum) {
    if (++count == 1) {//第一个,因此平均值就是本身
      average = datum;
    } else {//因为不是第一个了,因此要计算一下平均值
      /**
       * 算法
       * average * (count - 1) 平均值*以前的count = 以前的总值
       * average * (count - 1) / count 表示以前的总值在现在的平均值
       * datum / count 表示本次添加的平均值,因此就是总平均值
       */
      average = average * (count - 1) / count + datum / count;   
    }
  }
  
  /**
   * @param datum
   *          item to remove to the running average
   * @throws IllegalStateException
   *           if count is 0
   */
  @Override
  public synchronized void removeDatum(double datum) {
    if (count == 0) {
      throw new IllegalStateException();
    }
    if (--count == 0) {
      average = Double.NaN;
    } else {
      average = average * (count + 1) / count - datum / count;
    }
  }
  
  /**
   * @param delta
   *          amount by which to change a datum in the running average
   * @throws IllegalStateException
   *           if count is 0
   * 表示其中原来添加进来的元素,有变更,比如原来加入的是20,现在变成23,因此参数就是3,此时平均值如何算
   * 
   * 比如现在一共存在的数字是
   * (22+25+28+21)/4  现在参数是-3
   * 因此变成(22+25+28+21-3)/4,即(22+25+28+21)/4 + (-3/4)
   */
  @Override
  public synchronized void changeDatum(double delta) {
    if (count == 0) {
      throw new IllegalStateException();
    }
    average += delta / count;
  }
  
  @Override
  public synchronized int getCount() {
    return count;
  }
  
  @Override
  public synchronized double getAverage() {
    return average;
  }

  @Override
  public RunningAverage inverse() {
    return new InvertedRunningAverage(this);
  }
  
  @Override
  public synchronized String toString() {
    return String.valueOf(average);
  }
  
}
