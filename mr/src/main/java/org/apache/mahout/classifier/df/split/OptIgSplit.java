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

package org.apache.mahout.classifier.df.split;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataUtils;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * <p>Optimized implementation of IgSplit.优化的实现
 * 
 * This class can be used when the criterion variable is the categorical attribute.</p>
 * 优化在分类属性方面没有太大的优化,但是在数值属性方面,只最多获取16个属性进行运算,从中查找合适的值作为决策树,也通过这16个值计算信息熵,这16个值是所有值的均值
 * 
 * <p>This code was changed in MAHOUT-1419 to deal in sampled splits among numeric
 * features to fix a performance problem. To generate some synthetic data that exercises
 * the issue, try for example generating 4 features of Normal(0,1) values with a random
 * boolean 0/1 categorical feature. In Scala:</p>
 *
 * 以下是scala代码
 * {@code
 *  val r = new scala.util.Random()
 *  val pw = new java.io.PrintWriter("random.csv")
 *  (1 to 10000000).foreach(e =>
 *    pw.println(r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               (if (r.nextBoolean()) 1 else 0))
 *   )
 *   pw.close()
 * }
 */
@Deprecated
public class OptIgSplit extends IgSplit {

  private static final int MAX_NUMERIC_SPLITS = 16;

  @Override
  public Split computeSplit(Data data, int attr) {
    if (data.getDataset().isNumerical(attr)) {
      return numericalSplit(data, attr);
    } else {
      return categoricalSplit(data, attr);
    }
  }

  /**
   * Computes the split for a CATEGORICAL attribute
   * 计算分类类型的属性
   */
  private static Split categoricalSplit(Data data, int attr) {
    double[] values = data.values(attr).clone();//该属性对应的所有值--过滤重复了

    double[] splitPoints = chooseCategoricalSplitPoints(values);//对属性值进行排序

    int numLabels = data.getDataset().nblabels();//所有的标签数量
    int[][] counts = new int[splitPoints.length][numLabels];//二维数组,[ [1,4,7] ,[1,2,6] ] 第一个数组表示第一个属性 拥有每一个标签多少条数据
    int[] countAll = new int[numLabels];//每一个标签出现的数量,比如值是{1,4,7} 表示第一个标签有一条数据,第二个标签有4条数据

    computeFrequencies(data, attr, splitPoints, counts, countAll);//计算counts和countAll

    int size = data.size();//数据总数
    double hy = entropy(countAll, size); // H(Y) 计算这些标签总的期望熵
    
    double hyx = 0.0; // H(Y|X) //表示在该属性确定的情况下,熵
    
    double invDataSize = 1.0 / size;//数据越多,该值越小,即数据条数的倒数

    for (int index = 0; index < splitPoints.length; index++) {//循环每一个属性
      size = DataUtils.sum(counts[index]);//获取该属性对应的数据条数
      /**
       * 说明
       * size * invDataSize 表示该属性所占概率
       * entropy(counts[index], size) 表示该属性的标签的期望熵
       */
      hyx += size * invDataSize * entropy(counts[index], size);
    }

    double ig = hy - hyx;//信息增益
    return new Split(attr, ig);
  }

  //计算每一个属性对每一个标签的数量
  static void computeFrequencies(Data data,//数据内容
                                 int attr,//哪个属性
                                 double[] splitPoints,//分类属性对应的所有值(过滤重复 且 已经排序)
                                 int[][] counts,//二维数组,[ [1,4,7] ,[1,2,6] ] 第一个数组表示第一个属性 拥有每一个标签多少条数据
                                 int[] countAll) {//每一个标签出现的数量,比如值是{1,4,7} 表示第一个标签有一条数据,第二个标签有4条数据
    Dataset dataset = data.getDataset();

    for (int index = 0; index < data.size(); index++) {//循环每一条数据
      Instance instance = data.get(index);
      int label = (int) dataset.getLabel(instance);//获取该数据对应的标签
      double value = instance.get(attr);//该标签对应属性值
      int split = 0;
      
      //因为splitPoints是对属性值进行了排序,因此查看value > splitPoints[split] 可以知道该属性value是第几个属性
      //其实我觉得不用这一步,因为排序了,所以直接获取splitPoints[value][label]++就可以了,不太明白为什么要这么操作---现在知道原因了,因为该方法不是给分类属性用的,当是数值类型的时候,需要这一步处理,因为数值类型的只是从所有数据中抽取了16个元素而已
      while (split < splitPoints.length && value > splitPoints[split]) {
        split++;
      }
      if (split < splitPoints.length) {
        counts[split][label]++;//说明该属性在label标签上有一条数据
      } // Otherwise it's in the last split, which we don't need to count
      countAll[label]++;//说明该label有一条数据
    }
  }

  /**
   * Computes the best split for a NUMERICAL attribute
   * 计算数值类型的属性
   */
  static Split numericalSplit(Data data, int attr) {
    double[] values = data.values(attr).clone();//获取该属性的所有值
    Arrays.sort(values);//对所有的值进行排序

    double[] splitPoints = chooseNumericSplitPoints(values);//选择出最多16个元素,并且元素的数量不大于16个,防止计算过多属性值,这16个都是平均分布的属性值

    int numLabels = data.getDataset().nblabels();//所有标签
    int[][] counts = new int[splitPoints.length][numLabels];//二维数组,[ [1,4,7] ,[1,2,6] ] 第一个数组表示第一个属性 拥有每一个标签多少条数据
    int[] countAll = new int[numLabels];//每一个标签出现的数量,比如值是{1,4,7} 表示第一个标签有一条数据,第二个标签有4条数据---表示全部数据
    int[] countLess = new int[numLabels];//每一个标签出现的数量,比如值是{1,4,7} 表示第一个标签有一条数据,第二个标签有4条数据---只是此时表示小于一个value值时候为前提

    computeFrequencies(data, attr, splitPoints, counts, countAll);//计算counts和countAll

    int size = data.size();
    double hy = entropy(countAll, size);//计算这些标签总的期望熵
    
    double invDataSize = 1.0 / size;//数据越多,该值越小,即数据条数的倒数

    int best = -1;//最好的值所在的index
    double bestIg = -1.0;//最好的信息增益

    // try each possible split value
    for (int index = 0; index < splitPoints.length; index++) {//循环每一个选择的属性值
      double ig = hy; //最终该属性值对应的信息增益

      DataUtils.add(countLess, counts[index]);//将counts[index]中每一个标签有多少条数据添加到countLess数组中,即小于该index对应的value值对应的标签数据
      DataUtils.dec(countAll, counts[index]);//从all中减去每一个标签有多少条数据

      // instance with attribute value < values[index]
      //表示该index对应的value的左子树,即小于该value的标签记录数
      size = DataUtils.sum(countLess);
      ig -= size * invDataSize * entropy(countLess, size);
      
      // instance with attribute value >= values[index]
      size = DataUtils.sum(countAll);
      ig -= size * invDataSize * entropy(countAll, size);

      if (ig > bestIg) {
        bestIg = ig;
        best = index;
      }
    }

    if (best == -1) {
      throw new IllegalStateException("no best split found !");
    }
    return new Split(attr, bestIg, splitPoints[best]);
  }

  /**
   * @return an array of values to split the numeric feature's values on when
   *  building candidate splits. When input size is <= MAX_NUMERIC_SPLITS + 1, it will
   *  return the averages between success values as split points. When larger, it will
   *  return MAX_NUMERIC_SPLITS approximate percentiles through the data.
   *  参数values是已经排序好的数据
   */
  private static double[] chooseNumericSplitPoints(double[] values) {
    if (values.length <= 1) {
      return values;
    }
    if (values.length <= MAX_NUMERIC_SPLITS + 1) {//说明values的长度小于16个,那么就正常计算即可
      double[] splitPoints = new double[values.length - 1];
      for (int i = 1; i < values.length; i++) {//从1开始,因为从0开始计算的话,没办法计算i-1的下标
        splitPoints[i-1] = (values[i] + values[i-1]) / 2.0;//获取两个数值的均值
      }
      return splitPoints;
    }
    
    //说明此时value的长度>16个,返回均匀的分配的16个指标
    Percentile distribution = new Percentile();
    distribution.setData(values);
    double[] percentiles = new double[MAX_NUMERIC_SPLITS];//返回只要16个
    for (int i = 0 ; i < percentiles.length; i++) {
      double p = 100.0 * ((i + 1.0) / (MAX_NUMERIC_SPLITS + 1.0));
      percentiles[i] = distribution.evaluate(p);
    }
    return percentiles;
  }

  //参数是该属性对应的所有值--过滤重复了
  //对数据进行排序
  private static double[] chooseCategoricalSplitPoints(double[] values) {
    // There is no great reason to believe that categorical value order matters,
    // but the original code worked this way, and it's not terrible in the absence
    // of more sophisticated analysis
	//对分类进行排序
    Collection<Double> uniqueOrderedCategories = new TreeSet<Double>();
    for (double v : values) {
      uniqueOrderedCategories.add(v);
    }
    
    double[] uniqueValues = new double[uniqueOrderedCategories.size()];
    Iterator<Double> it = uniqueOrderedCategories.iterator();
    for (int i = 0; i < uniqueValues.length; i++) {
      uniqueValues[i] = it.next();
    }
    return uniqueValues;
  }

  /**
   * Computes the Entropy
   * 计算熵
   *
   * @param counts   counts[i] = numInstances with label i
   * @param dataSize numInstances
   */
  private static double entropy(int[] counts, int dataSize) {
    if (dataSize == 0) {
      return 0.0;
    }

    double entropy = 0.0;

    for (int count : counts) {
      if (count > 0) {
        double p = count / (double) dataSize; //表示该标签所占比例,即概率
        entropy -= p * Math.log(p);
      }
    }

    return entropy / LOG2;
  }

}
