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
import org.apache.mahout.classifier.df.data.conditions.Condition;

import java.util.Arrays;

/**
 * Default, not optimized, implementation of IgSplit
 */
@Deprecated
public class DefaultIgSplit extends IgSplit {
  
  /** used by entropy() 熵 */
  private int[] counts;//有多少个label属性,即每一个label属性出现了多少次
  
  //计算一个属性对应的信息增益
  @Override
  public Split computeSplit(Data data, int attr) {
    if (data.getDataset().isNumerical(attr)) {//该属性是数值类型的
      double[] values = data.values(attr);//获取给定属性的值---过滤重复
      double bestIg = -1;//最好的split时候,真实值作为参数 得到的ig分数,即信息增益
      double bestSplit = 0.0;//最好的split时候,是什么真实的值
      
      for (double value : values) {
        double ig = numericalIg(data, attr, value);//计算信息增益,信息增益越大,说明该分支越确定.因此越应该是用该属性作为决策树
        if (ig > bestIg) {
          bestIg = ig;
          bestSplit = value;
        }
      }
      
      return new Split(attr, bestIg, bestSplit);
    } else {//分类属性 计算信息增益
      double ig = categoricalIg(data, attr);
      
      return new Split(attr, ig);//不明白为什么这个时候没有计算哪个分类值对应的信息增益最大,而只是计算了该属性整体占用的信息增益
    }
  }
  
  /**
   * Computes the Information Gain for a CATEGORICAL attribute
   * 分类属性 计算信息增益
   */
  double categoricalIg(Data data, int attr) {
    double[] values = data.values(attr);//获取该属性的所有分类 ---过滤重复
    
    double hy = entropy(data); // H(Y) //计算该数据集合总的期望熵
    
    double hyx = 0.0; // H(Y|X)
    
    double invDataSize = 1.0 / data.size();//数据越多,该值越小,即数据条数的倒数
    
    for (double value : values) {//循环每一个分类
      Data subset = data.subset(Condition.equals(attr, value));//获取该分类下所有的数据子集
      /**
       * 说明:
       * subset.size() * invDataSize 表示该分类所占比例,即概率p
       * entropy(subset)表示该分类的子集占用多少期望熵
       * 因此该值就是该分类占用多少期望,所有分类之和就是熵的期望和
       */
      hyx += subset.size() * invDataSize * entropy(subset);
    }
    
    return hy - hyx;//这就是信息增益
  }
  
  /**
   * Computes the Information Gain for a NUMERICAL attribute given a splitting value
   * 为数值类型的属性   计算信息增益
   * 
   * 即属性attr的数值为split时候,信息增益是多少
   */
  double numericalIg(Data data, int attr, double split) {
    double hy = entropy(data);//计算标签对应的熵的期望
    
    double invDataSize = 1.0 / data.size();//数据越多,该值越小,即数据条数的倒数
    
    // LO subset
    Data subset = data.subset(Condition.lesser(attr, split));//缩小数据--左子树
    
    //下面两个方法并不是递归,只是简单的计算一次
    /**
     * 说明
     * subset.size() * invDataSize 表示左子树的占有概率
     * p*entropy(subset) 表示 左子树的期望信息熵
     * 用hy该标签的总信息熵-左边树的信息熵,就是信息增益
     */
    hy -= subset.size() * invDataSize * entropy(subset);
    
    // HI subset
    subset = data.subset(Condition.greaterOrEquals(attr, split));//缩小数据--右子树
    hy -= subset.size() * invDataSize * entropy(subset);
    
    return hy;
  }
  
  /**
   * Computes the Entropy
   * 计算信息熵---即计算每一个标签对应的熵的期望
   * 
   * 每一次经过决策树的处理,原始数据data已经在不断的缩小了,因此每次计算的信息熵是不一样的,因为数据源是不一样的了
   */
  protected double entropy(Data data) {
    double invDataSize = 1.0 / data.size();//数据越多,该值越小,即数据条数的倒数
    
    if (counts == null) {
      counts = new int[data.getDataset().nblabels()];//返回有多少个label属性
    }
    
    Arrays.fill(counts, 0);
    data.countLabels(counts);//计算每一个标签对应多少条数据
    
    double entropy = 0.0; //信息熵就是概率的期望
    for (int label = 0; label < data.getDataset().nblabels(); label++) {//循环每一个label
      int count = counts[label];//该label出现了多少次
      if (count == 0) {
        continue; // otherwise we get a NaN
      }
      double p = count * invDataSize;//得到该label的行数占总函数的比例,即概率
      /**
       * 备注:
       * math.log(p) / Math.log(2.0) 可以通过换底公式,转换成任意底 log2^p,即最终转换成以2为底,求p的log
       * 
			System.out.println(Math.log(0.3));//-1.2039728043259361
			System.out.println(Math.log(2));//0.6931471805599453
			System.out.println(Math.log(0.3)/Math.log(2));//-1.7369655941662063
       */
      entropy += -p * Math.log(p) / LOG2; 
    }
    
    return entropy;
  }
  
}
