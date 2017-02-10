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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.df.data.Instance;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Represents an abstract node of a decision tree
 */
@Deprecated
public abstract class Node implements Writable {
  
  protected enum Type {
    LEAF,//叶子节点,即真正的label了
    NUMERICAL,//数字形式
    CATEGORICAL//类别形式,即离散的分类中选择一个
  }
  
  /**
   * predicts the label for the instance
   * 预测该实例的一个label
   * 
   * @return -1 if the label cannot be predicted,-1说明这个实例不能被预测一个label
   */
  public abstract double classify(Instance instance);
  
  /**
   * @return the total number of nodes of the tree 返回该节点组成的树一共包括自己 有多少个节点
   */
  public abstract long nbNodes();
  
  /**
   * @return the maximum depth of the tree
   */
  public abstract long maxDepth();
  
  protected abstract Type getType();
  
  public static Node read(DataInput in) throws IOException {
    Type type = Type.values()[in.readInt()];
    Node node;
    
    switch (type) {
      case LEAF:
        node = new Leaf();
        break;
      case NUMERICAL:
        node = new NumericalNode();
        break;
      case CATEGORICAL:
        node = new CategoricalNode();
        break;
      default:
        throw new IllegalStateException("This implementation is not currently supported");
    }
    
    node.readFields(in);
    
    return node;
  }
  
  @Override
  public final String toString() {
    return getType() + ":" + getString() + ';';
  }
  
  protected abstract String getString();
  
  @Override
  public final void write(DataOutput out) throws IOException {
    out.writeInt(getType().ordinal());//写入Node类型
    writeNode(out);
  }
  
  //具体怎么为Node赋值,让子类去自己实现
  protected abstract void writeNode(DataOutput out) throws IOException;
  
}
