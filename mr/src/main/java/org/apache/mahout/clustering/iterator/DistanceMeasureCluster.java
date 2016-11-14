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

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * 表示如果对一个中心点进行编辑距离计算
 */
public class DistanceMeasureCluster extends AbstractCluster {

  private DistanceMeasure measure;

  public DistanceMeasureCluster(Vector point, int id, DistanceMeasure measure) {
    super(point, id);
    this.measure = measure;
  }

  public DistanceMeasureCluster() {
  }

  @Override
  public void configure(Configuration job) {
    if (measure != null) {
      measure.configure(job);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    String dm = in.readUTF();
    this.measure = ClassUtils.instantiateAs(dm, DistanceMeasure.class);
    super.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(measure.getClass().getName());
    super.write(out);
  }

  /**
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
  @Override
  public double pdf(VectorWritable vw) {
    return 1 / (1 + measure.distance(vw.get(), getCenter()));
  }

  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return new DistanceMeasureCluster(getCenter(), getId(), measure);
  }

  public DistanceMeasure getMeasure() {
    return measure;
  }

  /**
   * @param measure
   *          the measure to set
   */
  public void setMeasure(DistanceMeasure measure) {
    this.measure = measure;
  }

  @Override
  public String getIdentifier() {
    return "DMC:" + getId();
  }

}
