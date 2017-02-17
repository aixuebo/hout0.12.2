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

package org.apache.mahout.classifier.df;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataUtils;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.node.Node;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Represents a forest of decision trees.
 * 代表一个森林决策树,即决策森林
 */
@Deprecated
public class DecisionForest implements Writable {
  
  private final List<Node> trees;//每一个Node都是一棵树,即例如DecisionTreeBuilder对象
  
  private DecisionForest() {
    trees = new ArrayList<>();
  }
  
  public DecisionForest(List<Node> trees) {
    Preconditions.checkArgument(trees != null && !trees.isEmpty(), "trees argument must not be null or empty");

    this.trees = trees;
  }
  
  List<Node> getTrees() {
    return trees;
  }

  /**
   * Classifies the data and calls callback for each classification
   * 对所有的数据进行预测,每一颗树都给与一个推荐的标签序号
   * 参数predictions第一个数组表示多少条数据,有多少条数据,就有多少个数组,第二层数组表示该第一条数据在每一个决策树中的最终决策标签序号
   */
  public void classify(Data data, double[][] predictions) {
    Preconditions.checkArgument(data.size() == predictions.length, "predictions.length must be equal to data.size()");

    if (data.isEmpty()) {
      return; // nothing to classify
    }

    int treeId = 0;//表示当前操作的是第几颗树
    for (Node tree : trees) {//循环每一颗树
      for (int index = 0; index < data.size(); index++) {//循环每一行数据
        if (predictions[index] == null) {
          predictions[index] = new double[trees.size()];
        }
        predictions[index][treeId] = tree.classify(data.get(index));//该树对该数据进行预测
      }
      treeId++;
    }
  }
  
  /**
   * predicts the label for the instance
   * 
   * @param rng
   *          Random number generator, used to break ties randomly
   * @return NaN if the label cannot be predicted
   * 为某一个数据进行--返回综合考虑后的标签值
   */
  public double classify(Dataset dataset, Random rng, Instance instance) {
    if (dataset.isNumerical(dataset.getLabelId())) {//该数据的标签是整数类型的,即回归问题---返回所有决策树的平均值
      double sum = 0;//所有决策树的总分数
      int cnt = 0;//有多少颗树对该记录有决策行为
      for (Node tree : trees) {//循环所有的决策树中每一颗树
        double prediction = tree.classify(instance);//返回该树对该记录的标签分数
        if (!Double.isNaN(prediction)) {
          sum += prediction;
          cnt++;
        }
      }

      if (cnt > 0) {
        return sum / cnt;//获取平均值
      } else {
        return Double.NaN;
      }
    } else {//该数据属于分类标签
      int[] predictions = new int[dataset.nblabels()];//有多少个label属性,即每一个label属性对应一个分数---即每一个label被多少个决策树打分了
      for (Node tree : trees) {
        double prediction = tree.classify(instance);//每一颗树都打一个分类
        if (!Double.isNaN(prediction)) {
          predictions[(int) prediction]++;
        }
      }

      if (DataUtils.sum(predictions) == 0) {//说明没有预测出来
        return Double.NaN; // no prediction available
      }

      return DataUtils.maxindex(rng, predictions);//找到values中最大值的index位置
    }
  }
  
  /**
   * @return Mean number of nodes per tree
   * 平均每一个树包含多少个节点
   */
  public long meanNbNodes() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.nbNodes();
    }
    
    return sum / trees.size();
  }
  
  /**
   * @return Total number of nodes in all the trees
   * 所有的树包含多少个节点
   */
  public long nbNodes() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.nbNodes();//返回该节点组成的树一共包括自己 有多少个节点
    }
    
    return sum;
  }
  
  /**
   * @return Mean maximum depth per tree
   * 平均最大深度
   */
  public long meanMaxDepth() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.maxDepth();
    }
    
    return sum / trees.size();
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof DecisionForest)) {
      return false;
    }
    
    DecisionForest rf = (DecisionForest) obj;
    
    return trees.size() == rf.getTrees().size() && trees.containsAll(rf.getTrees());
  }
  
  @Override
  public int hashCode() {
    return trees.hashCode();
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(trees.size());
    for (Node tree : trees) {
      tree.write(dataOutput);
    }
  }

  /**
   * Reads the trees from the input and adds them to the existing trees
   */
  @Override
  public void readFields(DataInput dataInput) throws IOException {
    int size = dataInput.readInt();
    for (int i = 0; i < size; i++) {
      trees.add(Node.read(dataInput));
    }
  }

  /**
   * Read the forest from inputStream
   * @param dataInput - input forest
   * @return {@link org.apache.mahout.classifier.df.DecisionForest}
   * @throws IOException
   */
  public static DecisionForest read(DataInput dataInput) throws IOException {
    DecisionForest forest = new DecisionForest();
    forest.readFields(dataInput);
    return forest;
  }

  /**
   * Load the forest from a single file or a directory of files
   * @throws java.io.IOException
   */
  public static DecisionForest load(Configuration conf, Path forestPath) throws IOException {
    FileSystem fs = forestPath.getFileSystem(conf);
    Path[] files;
    if (fs.getFileStatus(forestPath).isDir()) {
      files = DFUtils.listOutputFiles(fs, forestPath);
    } else {
      files = new Path[]{forestPath};
    }

    DecisionForest forest = null;
    for (Path path : files) {
      try (FSDataInputStream dataInput = new FSDataInputStream(fs.open(path))) {
        if (forest == null) {
          forest = read(dataInput);
        } else {
          forest.readFields(dataInput);
        }
      }
    }

    return forest;
    
  }

}
