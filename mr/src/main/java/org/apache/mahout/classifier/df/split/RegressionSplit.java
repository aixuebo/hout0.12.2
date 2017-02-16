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

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Instance;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

/**
 * Regression problem implementation of IgSplit. This class can be used when the criterion variable is the numerical
 * attribute.
 * 回归问题的实现
 */
@Deprecated
public class RegressionSplit extends IgSplit {
  
  /**
   * Comparator for Instance sort
   * 比较两条记录,两条记录按照属性值进行排序
   */
  private static class InstanceComparator implements Comparator<Instance>, Serializable {
    private final int attr;

    InstanceComparator(int attr) {
      this.attr = attr;
    }
    
    @Override
    public int compare(Instance arg0, Instance arg1) {
      return Double.compare(arg0.get(attr), arg1.get(attr));
    }
  }
  
  //计算信息增益
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
   * 对分类属性进行计算信息增益
   */
  private static Split categoricalSplit(Data data, int attr) {
	  
    FullRunningAverage[] ra = new FullRunningAverage[data.getDataset().nbValues(attr)];//该分类属性有多少个分类值,就有多少个数组,即每一个具体的分类值,对应一个该FullRunningAverage对象
    for (int i = 0; i < ra.length; i++) {
        ra[i] = new FullRunningAverage();
    }
    
    double[] sk = new double[data.getDataset().nbValues(attr)];//该分类属性有多少个分类值,就有多少个数组,即每一个具体的分类值,对应一个double值,表示该分类值对应的信息熵
    
    FullRunningAverage totalRa = new FullRunningAverage();//不考虑分类值,所有标签值
    
    double totalSk = 0.0;//总的信息熵

    for (int i = 0; i < data.size(); i++) {//循环每一条数据
      // computes the variance
      Instance instance = data.get(i);
      int value = (int) instance.get(attr);//该属性对应的值
      double xk = data.getDataset().getLabel(instance);//该数据对应的标签
      if (ra[value].getCount() == 0) {//获取该分类值对应的FullRunningAverage 添加元素
        ra[value].addDatum(xk);
        sk[value] = 0.0;
      } else {
        double mk = ra[value].getAverage();//获取此时的平均值(标签)
        ra[value].addDatum(xk);//添加该标签对应的值
        sk[value] += (xk - mk) * (xk - ra[value].getAverage());
      }

      // total variance
      if (i == 0) {
        totalRa.addDatum(xk);
        totalSk = 0.0;
      } else {
        double mk = totalRa.getAverage();
        totalRa.addDatum(xk);
        totalSk += (xk - mk) * (xk - totalRa.getAverage());
      }
    }

    // computes the variance gain
    //获取信息增益
    double ig = totalSk;
    for (double aSk : sk) {
      ig -= aSk;
    }

    return new Split(attr, ig);
  }
  
  /**
   * Computes the best split for a NUMERICAL attribute
   * 对数值属性进行计算信息增益
   */
  private static Split numericalSplit(Data data, int attr) {
    FullRunningAverage[] ra = new FullRunningAverage[2];
    for (int i = 0; i < ra.length; i++) {
      ra[i] = new FullRunningAverage();
    }

    // Instance sort 对数据内容进行排序,按照属性值进行排序
    Instance[] instances = new Instance[data.size()];
    for (int i = 0; i < data.size(); i++) {
      instances[i] = data.get(i);
    }
    Arrays.sort(instances, new InstanceComparator(attr));

    
    double[] sk = new double[2];//对应的信息熵,左边和右边两侧对应的信息熵
    
    for (Instance instance : instances) {//循环排序后的数据
      double xk = data.getDataset().getLabel(instance);//该数据对应的标签
      if (ra[1].getCount() == 0) {
        ra[1].addDatum(xk);
        sk[1] = 0.0;
      } else {
        double mk = ra[1].getAverage();
        ra[1].addDatum(xk);
        sk[1] += (xk - mk) * (xk - ra[1].getAverage());
      }
    }
    
    double totalSk = sk[1];//总的信息熵

    // find the best split point 获取最好的属性值点
    double preSplit = Double.NaN;//上一次的属性对应的具体值
    double split = Double.NaN;//计算后的属性值
    double bestVal = Double.MAX_VALUE;//最好的属性值
    double bestSk = 0.0;//最好的属性对应的信息熵

    // computes total variance
    for (Instance instance : instances) {//循环排序后的数据
      double xk = data.getDataset().getLabel(instance);//获取标签值

      if (instance.get(attr) > preSplit) {
        double curVal = sk[0] / ra[0].getCount() + sk[1] / ra[1].getCount();//计算属性值--表示左右树的熵和
        if (curVal < bestVal) {
          bestVal = curVal;
          bestSk = sk[0] + sk[1];
          split = (instance.get(attr) + preSplit) / 2.0;//属性值是当前属性值与preSplit的均值
        }
      }

      //计算该标签
      // computes the variance
      if (ra[0].getCount() == 0) {
        ra[0].addDatum(xk);
        sk[0] = 0.0;
      } else {
        double mk = ra[0].getAverage();
        ra[0].addDatum(xk);
        sk[0] += (xk - mk) * (xk - ra[0].getAverage());
      }

      //刨除该标签
      double mk = ra[1].getAverage();
      ra[1].removeDatum(xk);
      sk[1] -= (xk - mk) * (xk - ra[1].getAverage());

      preSplit = instance.get(attr);
    }

    // computes the variance gain
    double ig = totalSk - bestSk;//计算信息增益

    return new Split(attr, ig, split);
  }
}
