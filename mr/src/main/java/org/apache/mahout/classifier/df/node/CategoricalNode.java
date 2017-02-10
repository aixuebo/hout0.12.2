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

package org.apache.mahout.classifier.df.node;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.data.Instance;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
@Deprecated
public class CategoricalNode extends Node {

  private int attr;//该属性属于向量的第几个属性
  private double[] values;//分类标签对应的数字
  private Node[] childs;//每一个分类对应的子节点
  
  public CategoricalNode() {
  }
  
  public CategoricalNode(int attr, double[] values, Node[] childs) {
    this.attr = attr;
    this.values = values;
    this.childs = childs;
  }
  
  @Override
  public double classify(Instance instance) {
    int index = ArrayUtils.indexOf(values, instance.get(attr));//在values中查找 instance.get(attr)对应的位置
    if (index == -1) {
      // value not available, we cannot predict
      return Double.NaN;
    }
    return childs[index].classify(instance);//获取一个分类,到一个分类下进一步计算
  }
  
  //最大深度就是1+子类中最大的深度
  @Override
  public long maxDepth() {
    long max = 0;
    
    for (Node child : childs) {
      long depth = child.maxDepth();
      if (depth > max) {
        max = depth;
      }
    }
    
    return 1 + max;
  }
  
  @Override
  public long nbNodes() {//该节点组成的tree,包括自己有多少个节点
    long nbNodes = 1;
    
    for (Node child : childs) {
      nbNodes += child.nbNodes();
    }
    
    return nbNodes;
  }
  
  @Override
  protected Type getType() {
    return Type.CATEGORICAL;
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof CategoricalNode)) {
      return false;
    }
    
    CategoricalNode node = (CategoricalNode) obj;
    
    return attr == node.attr && Arrays.equals(values, node.values) && Arrays.equals(childs, node.childs);
  }
  
  @Override
  public int hashCode() {
    int hashCode = attr;
    for (double value : values) {
      hashCode = 31 * hashCode + (int) Double.doubleToLongBits(value);
    }
    for (Node node : childs) {
      hashCode = 31 * hashCode + node.hashCode();
    }
    return hashCode;
  }
  
  @Override
  protected String getString() {
    StringBuilder buffer = new StringBuilder();
    
    for (Node child : childs) {
      buffer.append(child).append(',');
    }
    
    return buffer.toString();
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    attr = in.readInt();
    values = DFUtils.readDoubleArray(in);
    childs = DFUtils.readNodeArray(in);
  }
  
  @Override
  protected void writeNode(DataOutput out) throws IOException {
    out.writeInt(attr);
    DFUtils.writeArray(out, values);
    DFUtils.writeArray(out, childs);
  }
}
