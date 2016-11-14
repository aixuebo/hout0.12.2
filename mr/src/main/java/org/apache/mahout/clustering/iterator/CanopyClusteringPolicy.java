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
package org.apache.mahout.clustering.iterator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

@Deprecated
public class CanopyClusteringPolicy extends AbstractClusteringPolicy {

  private double t1;
  private double t2;

  @Override
  public Vector select(Vector probabilities) {
    int maxValueIndex = probabilities.maxValueIndex();//最大概率分数的下标
    Vector weights = new SequentialAccessSparseVector(probabilities.size());//创新一个新的向量,是疏松向量,只有值的下标才会在该向量里面存储值,此时是空的
    weights.set(maxValueIndex, 1.0);//向该疏松向量添加值,仅添加了一个值,表示哪个下标位置的概率为1
    return weights;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(t1);
    out.writeDouble(t2);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.t1 = in.readDouble();
    this.t2 = in.readDouble();
  }
  
}
