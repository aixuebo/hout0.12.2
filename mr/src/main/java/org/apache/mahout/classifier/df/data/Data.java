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

package org.apache.mahout.classifier.df.data;

import org.apache.mahout.classifier.df.data.conditions.Condition;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

/**
 * Holds a list of vectors and their corresponding Dataset. contains various operations that deals with the
 * vectors (subset, count,...)
 * 
 */
@Deprecated
public class Data implements Cloneable {
  
  private final List<Instance> instances;//数据内容
  
  private final Dataset dataset;//数据集合.表示数据的title等信息描述

  public Data(Dataset dataset) {
    this.dataset = dataset;
    this.instances = new ArrayList<>();
  }

  public Data(Dataset dataset, List<Instance> instances) {
    this.dataset = dataset;
    this.instances = new ArrayList<>(instances);
  }
  
  /**
   * @return the number of elements
   */
  public int size() {
    return instances.size();
  }
  
  /**
   * @return true if this data contains no element
   */
  public boolean isEmpty() {
    return instances.isEmpty();
  }
  
  /**
   * @param v
   *          element whose presence in this list if to be searched
   * @return true is this data contains the specified element.
   */
  public boolean contains(Instance v) {
    return instances.contains(v);
  }

    /**
   * Returns the element at the specified position
   * 
   * @param index
   *          index of element to return
   * @return the element at the specified position
   * @throws IndexOutOfBoundsException
   *           if the index is out of range
   */
  public Instance get(int index) {
    return instances.get(index);
  }
  
  /**
   * @return the subset from this data that matches the given condition
   */
  public Data subset(Condition condition) {
    List<Instance> subset = new ArrayList<>();
    
    for (Instance instance : instances) {
      if (condition.isTrueFor(instance)) {
        subset.add(instance);
      }
    }
    
    return new Data(dataset, subset);
  }

   /**
   * if data has N cases, sample N cases at random -but with replacement.可能有代替
   * 即对原始的数据内容进行打乱顺序,比如原来有100条记录,最终结果是随机产生100条记录,这100条记录可能是有重复的数据
   * 
   * 抽样数据
   */
  public Data bagging(Random rng) {
    int datasize = size();
    List<Instance> bag = new ArrayList<>(datasize);
    
    for (int i = 0; i < datasize; i++) {
      bag.add(instances.get(rng.nextInt(datasize)));
    }
    
    return new Data(dataset, bag);
  }
  
  /**
   * if data has N cases, sample N cases at random -but with replacement.
   * 
   * @param sampled
   *          indicating which instance has been sampled 数组用于表示哪些数据被抽样带走了,true表示被抽样带走了
   * 
   * @return sampled data
   * 抽样数据
   */
  public Data bagging(Random rng, boolean[] sampled) {
    int datasize = size();//总数据大小
    List<Instance> bag = new ArrayList<>(datasize);
    
    for (int i = 0; i < datasize; i++) {
      int index = rng.nextInt(datasize);//随机产生一个位置
      bag.add(instances.get(index));//添加该位置到包中
      sampled[index] = true;
    }
    
    return new Data(dataset, bag);
  }
  
  /**
   * Splits the data in two, returns one part, and this gets the rest of the data. <b>VERY SLOW!</b>
   * 非常慢的操作,会将数据拆分成两组,这个函数选择抽取出subsize个数据
   */
  public Data rsplit(Random rng, int subsize) {
    List<Instance> subset = new ArrayList<>(subsize);
    
    for (int i = 0; i < subsize; i++) {
      subset.add(instances.remove(rng.nextInt(instances.size())));
    }
    
    return new Data(dataset, subset);
  }
  
  /**
   * checks if all the vectors have identical attribute values
   * 
   * @return true is all the vectors are identical or the data is empty<br>
   *         false otherwise
   * true表示所有的数据有相同的内容
   */
  public boolean isIdentical() {
    if (isEmpty()) {
      return true;
    }
    
    Instance instance = get(0);
    for (int attr = 0; attr < dataset.nbAttributes(); attr++) {//循环每一个属性
      for (int index = 1; index < size(); index++) {//针对每一个属性,都判断所有元素是否在该属性上有相同的数据
        if (get(index).get(attr) != instance.get(attr)) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  /**
   * checks if all the vectors have identical label values
   * true表示所有的数据对应的标签都是相同的,即只有一个标签内容
   */
  public boolean identicalLabel() {
    if (isEmpty()) {
      return true;
    }
    
    double label = dataset.getLabel(get(0));
    for (int index = 1; index < size(); index++) {
      if (dataset.getLabel(get(index)) != label) {
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * finds all distinct values of a given attribute
   * 获取给定属性的值---过滤重复
   */
  public double[] values(int attr) {
    Collection<Double> result = new HashSet<>();//不重复的Set
    
    for (Instance instance : instances) {
      result.add(instance.get(attr));//将每一条记录对应该属性的值添加到set中
    }
    
    double[] values = new double[result.size()];
    
    int index = 0;
    for (Double value : result) {
      values[index++] = value;
    }
    
    return values;
  }
  
  @Override
  public Data clone() {
    return new Data(dataset, new ArrayList<>(instances));
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Data)) {
      return false;
    }
    
    Data data = (Data) obj;
    
    return instances.equals(data.instances) && dataset.equals(data.dataset);
  }
  
  @Override
  public int hashCode() {
    return instances.hashCode() + dataset.hashCode();
  }
  
  /**
   * extract the labels of all instances
   * 抽取所有数据中标签这一列
   */
  public double[] extractLabels() {
    double[] labels = new double[size()];//所有数据,每一个数据对应的标签内容
    
    for (int index = 0; index < labels.length; index++) {
      labels[index] = dataset.getLabel(get(index));
    }
    
    return labels;
  }

    /**
   * finds the majority label, breaking ties randomly<br>
   * This method can be used when the criterion variable is the categorical attribute.
   *
   * @return the majority label value
   * 找出主要的标签所在index
   */
  public int majorityLabel(Random rng) {
    // count the frequency of each label value
    int[] counts = new int[dataset.nblabels()];//计算每一个标签对应多少条数据
    
    for (int index = 0; index < size(); index++) {
      counts[(int) dataset.getLabel(get(index))]++;
    }
    
    // find the label values that appears the most
    return DataUtils.maxindex(rng, counts);//找到标签数据最多的标签所在index
  }
  
  /**
   * Counts the number of occurrences of each label value<br>
   * This method can be used when the criterion variable is the categorical attribute.
   * 
   * @param counts
   *          will contain the results, supposed to be initialized at 0
   * 计算每一个标签对应多少条数据
   */
  public void countLabels(int[] counts) {
    for (int index = 0; index < size(); index++) {
      counts[(int) dataset.getLabel(get(index))]++;
    }
  }
  
  public Dataset getDataset() {
    return dataset;
  }
}
