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
import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

public abstract class AbstractClusteringPolicy implements ClusteringPolicy {
  
  @Override
  public abstract void write(DataOutput out) throws IOException;
  
  @Override
  public abstract void readFields(DataInput in) throws IOException;
  
  @Override
  public Vector select(Vector probabilities) {
    int maxValueIndex = probabilities.maxValueIndex();//最大概率分数的下标
    Vector weights = new SequentialAccessSparseVector(probabilities.size());//创新一个新的向量,是疏松向量,只有值的下标才会在该向量里面存储值,此时是空的
    weights.set(maxValueIndex, 1.0);//向该疏松向量添加值,仅添加了一个值,表示哪个下标位置的概率为1
    return weights;
  }
  
  @Override
  public void update(ClusterClassifier posterior) {
    // nothing to do in general here
  }
  
  //返回该参数向量在中心向量中打分占比,返回向量是一个打分值,分数越大,说明越属于该中心分类
  @Override
  public Vector classify(Vector data, ClusterClassifier prior) {
    List<Cluster> models = prior.getModels();
    int i = 0;
    Vector pdfs = new DenseVector(models.size());//保存每一个中心点model 与 data向量的pdf距离
    for (Cluster model : models) {
      pdfs.set(i++, model.pdf(new VectorWritable(data)));//计算每一个中心点model 与 data向量的pdf距离
    }
    /**
     * 每一个打分的值*(1/总分) = 相当于对每一个分转换成占比了,
     * 比如原来向量有三个分,0.15,0.35,0.2
     * 因此最终是0.15/0.7,0.35/0.7,0.2/0.7,因此是占比0.21,0.5,0.29
     */
    return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
  }
  
  //重新计算每一个分类的中心值
  @Override
  public void close(ClusterClassifier posterior) {
    for (Cluster cluster : posterior.getModels()) {
      cluster.computeParameters();
    }
    
  }
  
}
