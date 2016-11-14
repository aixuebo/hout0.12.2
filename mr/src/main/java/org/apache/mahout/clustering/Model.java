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

package org.apache.mahout.clustering;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VectorWritable;

/**
 * A model is a probability distribution over observed data points and allows
 * the probability of any data point to be computed. All Models have a
 * persistent representation and extend
 * WritablesampleFromPosterior(Model<VectorWritable>[])
 */
public interface Model<O> extends Writable {
  
  /**
   * Return the probability that the observation is described by this model
   * 返回一个概率
   * @param x
   *          an Observation from the posterior
   * @return the probability that x is in the receiver
   * 返回该元素属于该模型的概率
   * 参见org.apache.mahout.clustering.iterator.DistanceMeasureCluster类的实现
   * 
   * DistanceMeasureCluster类备注
   * 返回该元素属于该模型的概率
   * 因为该类DistanceMeasureCluster本身就表示一个分类,因此该方法表示参数向量 属于 本类这个分类的可能性
   * 既然是可能性,那么就是一个概率,属于0-1之间的概率
   * 
   * 那么怎么转换成概率呢,
   * 已知是 两个向量之间的距离越大,说明越不是一个分类
   * 
   * 如果我们用1/距离,因此距离越大,说明值越小,这样就说明概率越小,但是分母距离又不能为0,因此分母用1+距离,刨除0带来的隐患。
   * 而且由于距离还可能是0-1之间分数,因此1/0.5 就变成大于100%的概率了,也不对,因此分母用1+距离也可以保证一定分母是大于1的整数,得到的概率一定是0-1之间,这样就非常完美了
   * 
   * 至于分母用1+距离,那么我用2+距离可以吗？答案是完全可以的
   */
  double pdf(O x);
  
  /**
   * Observe the given observation, retaining information about it
   * 
   * @param x
   *          an Observation from the posterior
   */
  void observe(O x);
  
  /**
   * Observe the given observation, retaining information about it
   * 
   * @param x
   *          an Observation from the posterior
   * @param weight
   *          a double weighting factor
   */
  void observe(O x, double weight);
  
  /**
   * Observe the given model, retaining information about its observations
   * 
   * @param x
   *          a Model<0>
   */
  void observe(Model<O> x);
  
  /**
   * Compute a new set of posterior parameters based upon the Observations that
   * have been observed since my creation
   */
  void computeParameters();
  
  /**
   * Return the number of observations that this model has seen since its
   * parameters were last computed
   * 
   * @return a long
   */
  long getNumObservations();
  
  /**
   * Return the number of observations that this model has seen over its
   * lifetime
   * 
   * @return a long
   */
  long getTotalObservations();
  
  /**
   * @return a sample of my posterior model
   */
  Model<VectorWritable> sampleFromPosterior();
  
}
