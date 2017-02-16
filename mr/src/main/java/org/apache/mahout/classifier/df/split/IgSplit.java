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

package org.apache.mahout.classifier.df.split;

import org.apache.mahout.classifier.df.data.Data;

/**
 * Computes the best split using the Information Gain measure
 * 使用信息增益测量,去计算最好的拆分方式
 */
@Deprecated
public abstract class IgSplit {
  
  static final double LOG2 = Math.log(2.0);//0.6931471805599453 ,即0.6931471805599453的2次方=e
  
  /**
   * Computes the best split for the given attribute
   * 给定一个属性.计算该属性的信息增益
   * 如果属性是数值类型的时候,还要获取哪个值对应的信息增益最大
   */
  public abstract Split computeSplit(Data data, int attr);
  
}
