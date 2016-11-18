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

/**
 * <p>
 * Interface for classes that can keep track of a running average of a series of numbers. One can add to or
 * remove from the series, as well as update a datum in the series. The class does not actually keep track of
 * the series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 * </p>
 * 计算平均值和count值,边运行边计算平均值和count
 */
public interface RunningAverage {
  
  /**
   * @param datum
   *          new item to add to the running average
   * @throws IllegalArgumentException
   *           if datum is {@link Double#NaN}
   * 添加一个double
   */
  void addDatum(double datum);
  
  /**
   * @param datum
   *          item to remove to the running average
   * @throws IllegalArgumentException
   *           if datum is {@link Double#NaN}
   * @throws IllegalStateException
   *           if count is 0
   * 减去一个数          
   */
  void removeDatum(double datum);
  
  /**
   * @param delta
   *          amount by which to change a datum in the running average
   * @throws IllegalArgumentException
   *           if delta is {@link Double#NaN}
   * @throws IllegalStateException
   *           if count is 0
   * 表示其中原来添加进来的元素,有变更,比如原来加入的是20,现在变成23,因此参数就是3,此时平均值如何算,看FullRunningAverage实现
   * 
   * 比如现在一共存在的数字是
   * (22+25+28+21)/4  现在参数是-3
   * 因此变成(22+25+28+21-3)/4,即(22+25+28+21)/4 + (-3/4)
   */
  void changeDatum(double delta);
  
  //count计数
  int getCount();
  
  //平均值
  double getAverage();

  /**
   * @return a (possibly immutable) object whose average is the negative of this object's
   */
  RunningAverage inverse();
  
}
