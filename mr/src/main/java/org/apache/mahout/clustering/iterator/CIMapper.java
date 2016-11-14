/*
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

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

public class CIMapper extends Mapper<WritableComparable<?>,VectorWritable,IntWritable,ClusterWritable> {
  
  private ClusterClassifier classifier;
  private ClusteringPolicy policy;

  //初始化代理对象和中心对象集合
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String priorClustersPath = conf.get(ClusterIterator.PRIOR_PATH_KEY);//读取中心节点存储的位置
    classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, new Path(priorClustersPath));//获取中心节点信息以及代理信息
    policy = classifier.getPolicy();
    policy.update(classifier);//更新代理信息
    super.setup(context);
  }

  //分配每一个元素所属聚类
  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context) throws IOException,
      InterruptedException {
    Vector probabilities = classifier.classify(value.get());//计算每一个分类的概率
    Vector selections = policy.select(probabilities);
    for (Element el : selections.nonZeroes()) {//查找非0的位置
      classifier.train(el.index(), value.get(), el.get());//该元素 属于第几个分类 、该元素向量原值、该元素对该分类的打分
    }
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
	  //TODO 好像有bug,这个mapper本身没有进行中心点重新计算过程
    List<Cluster> clusters = classifier.getModels();//获取所有的聚类组
    ClusterWritable cw = new ClusterWritable();
    for (int index = 0; index < clusters.size(); index++) {//循环每一个聚类
      cw.setValue(clusters.get(index));//设置第几个聚类
      context.write(new IntWritable(index), cw);
    }
    super.cleanup(context);
  }
  
}
