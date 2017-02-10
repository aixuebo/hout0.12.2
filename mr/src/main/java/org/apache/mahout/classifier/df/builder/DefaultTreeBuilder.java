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

package org.apache.mahout.classifier.df.builder;

import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.data.conditions.Condition;
import org.apache.mahout.classifier.df.node.CategoricalNode;
import org.apache.mahout.classifier.df.node.Leaf;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.classifier.df.node.NumericalNode;
import org.apache.mahout.classifier.df.split.IgSplit;
import org.apache.mahout.classifier.df.split.OptIgSplit;
import org.apache.mahout.classifier.df.split.Split;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Builds a Decision Tree <br>
 * Based on the algorithm described in the "Decision Trees" tutorials by Andrew W. Moore, available at:<br>
 * <br>
 * http://www.cs.cmu.edu/~awm/tutorials
 * <br><br>
 * This class can be used when the criterion variable is the categorical attribute.
 */
@Deprecated
public class DefaultTreeBuilder implements TreeBuilder {

  private static final Logger log = LoggerFactory.getLogger(DefaultTreeBuilder.class);

  private static final int[] NO_ATTRIBUTES = new int[0];

  /**
   * indicates which CATEGORICAL attributes have already been selected in the parent nodes
   * 每一个属性对应一个位置,true表示该位置是分类或者label属性
   */
  private boolean[] selected;
  /**
   * number of attributes to select randomly at each node
   */
  private int m = 1;
  /**
   * IgSplit implementation
   */
  private final IgSplit igSplit;

  public DefaultTreeBuilder() {
    igSplit = new OptIgSplit();
  }

  public void setM(int m) {
    this.m = m;
  }

  //根据数据的信息熵,将数据生成树
  @Override
  public Node build(Random rng, Data data) {

    if (selected == null) {
      selected = new boolean[data.getDataset().nbAttributes()];
      selected[data.getDataset().getLabelId()] = true; // never select the label设置label肯定是分类属性
    }

    if (data.isEmpty()) {
      return new Leaf(-1);
    }
    if (isIdentical(data)) {//说明数据都一样
      return new Leaf(data.majorityLabel(rng));
    }
    if (data.identicalLabel()) {//说明lable都一样,即就一个分类标签结果
      return new Leaf(data.getDataset().getLabel(data.get(0)));
    }

    int[] attributes = randomAttributes(rng, selected, m);//随机选择m个属性,返回这次选择的属性index集合
    if (attributes == null || attributes.length == 0) {
      // we tried all the attributes and could not split the data anymore
      return new Leaf(data.majorityLabel(rng));
    }

    // find the best split
    Split best = null;//计算最好的属性
    for (int attr : attributes) {
      Split split = igSplit.computeSplit(data, attr);//计算每一个属性最好的value是什么时候获取的熵
      if (best == null || best.getIg() < split.getIg()) {//设置最好的熵是哪个属性
        best = split;
      }
    }

    boolean alreadySelected = selected[best.getAttr()];//true表示最好的属性已经选择了
    if (alreadySelected) {
      // attribute already selected
      log.warn("attribute {} already selected in a parent node", best.getAttr());
    }

    Node childNode;
    if (data.getDataset().isNumerical(best.getAttr())) {//最好的属性是数值类型的
      boolean[] temp = null;

      Data loSubset = data.subset(Condition.lesser(best.getAttr(), best.getSplit()));
      Data hiSubset = data.subset(Condition.greaterOrEquals(best.getAttr(), best.getSplit()));

      if (loSubset.isEmpty() || hiSubset.isEmpty()) {
        // the selected attribute did not change the data, avoid using it in the child notes
        selected[best.getAttr()] = true;
      } else {
        // the data changed, so we can unselect all previousely selected NUMERICAL attributes
        temp = selected;
        selected = cloneCategoricalAttributes(data.getDataset(), selected);
      }

      Node loChild = build(rng, loSubset);
      Node hiChild = build(rng, hiSubset);

      // restore the selection state of the attributes
      if (temp != null) {
        selected = temp;
      } else {
        selected[best.getAttr()] = alreadySelected;
      }

      childNode = new NumericalNode(best.getAttr(), best.getSplit(), loChild, hiChild);
    } else { // CATEGORICAL attribute 最好的属性是分类类型的
      selected[best.getAttr()] = true;

      double[] values = data.values(best.getAttr());
      Node[] children = new Node[values.length];

      for (int index = 0; index < values.length; index++) {
        Data subset = data.subset(Condition.equals(best.getAttr(), values[index]));
        children[index] = build(rng, subset);
      }

      selected[best.getAttr()] = alreadySelected;

      childNode = new CategoricalNode(best.getAttr(), values, children);
    }

    return childNode;
  }

  /**
   * checks if all the vectors have identical attribute values. Ignore selected attributes.
   *
   * @return true is all the vectors are identical or the data is empty<br>
   *         false otherwise
   */
  private boolean isIdentical(Data data) {
    if (data.isEmpty()) {
      return true;
    }

    Instance instance = data.get(0);
    for (int attr = 0; attr < selected.length; attr++) {
      if (selected[attr]) {
        continue;
      }

      for (int index = 1; index < data.size(); index++) {
        if (data.get(index).get(attr) != instance.get(attr)) {
          return false;
        }
      }
    }

    return true;
  }


  /**
   * Make a copy of the selection state of the attributes, unselect all numerical attributes
   *
   * @param selected selection state to clone
   * @return cloned selection state
   */
  private static boolean[] cloneCategoricalAttributes(Dataset dataset, boolean[] selected) {
    boolean[] cloned = new boolean[selected.length];

    for (int i = 0; i < selected.length; i++) {
      cloned[i] = !dataset.isNumerical(i) && selected[i];
    }

    return cloned;
  }

  /**
   * Randomly selects m attributes to consider for split, excludes IGNORED and LABEL attributes
   *
   * @param rng      random-numbers generator
   * @param selected attributes' state (selected or not) 位置为true表示已经选择了
   * @param m        number of attributes to choose 要继续选择m个属性
   * @return list of selected attributes' indices, or null if all attributes have already been selected 返回这次选择的属性index集合
   */
  protected static int[] randomAttributes(Random rng, boolean[] selected, int m) {
    int nbNonSelected = 0; // number of non selected attributes
    for (boolean sel : selected) {
      if (!sel) {
        nbNonSelected++;
      }
    }

    if (nbNonSelected == 0) {
      log.warn("All attributes are selected !");
      return NO_ATTRIBUTES;
    }

    int[] result;
    if (nbNonSelected <= m) {//要去选择M个,发现没选择的比M要小.因此全部将没选择的作为选择处理
      // return all non selected attributes
      result = new int[nbNonSelected];
      int index = 0;
      for (int attr = 0; attr < selected.length; attr++) {
        if (!selected[attr]) {
          result[index++] = attr;
        }
      }
    } else {
      result = new int[m];//选择M个没有选择的值
      for (int index = 0; index < m; index++) {
        // randomly choose a "non selected" attribute
        int rind;
        do {
          rind = rng.nextInt(selected.length);//随机产生一个位置
        } while (selected[rind]);//一直到该位置为false,则退出

        result[index] = rind;
        selected[rind] = true; // temporarily set the chosen attribute to be selected  暂时将其设置为选择
      }

      // the chosen attributes are not yet selected
      for (int attr : result) {//因为已经暂时将没选择的设置为选择了,因此要重新设置成没选择
        selected[attr] = false;
      }
    }

    return result;
  }
}
