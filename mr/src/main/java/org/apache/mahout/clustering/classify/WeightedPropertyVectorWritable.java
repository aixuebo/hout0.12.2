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

package org.apache.mahout.clustering.classify;

import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.math.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * 表示一个带有权重的向量,因此包含一个向量 以及该向量对应的权重值
 * 同时还增加了一些额外的属性信息对象
 */
public class WeightedPropertyVectorWritable extends WeightedVectorWritable {

  private Map<Text, Text> properties;//属性信息对象

  public WeightedPropertyVectorWritable() {
  }

  public WeightedPropertyVectorWritable(Map<Text, Text> properties) {
    this.properties = properties;
  }

  public WeightedPropertyVectorWritable(double weight, Vector vector, Map<Text, Text> properties) {
    super(weight, vector);
    this.properties = properties;
  }

  public Map<Text, Text> getProperties() {
    return properties;
  }

  public void setProperties(Map<Text, Text> properties) {
    this.properties = properties;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int size = in.readInt();
    if (size > 0) {
      properties = new HashMap<>();
      for (int i = 0; i < size; i++) {
        Text key = new Text(in.readUTF());
        Text val = new Text(in.readUTF());
        properties.put(key, val);
      }
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(properties != null ? properties.size() : 0);
    if (properties != null) {
      for (Map.Entry<Text, Text> entry : properties.entrySet()) {
        out.writeUTF(entry.getKey().toString());
        out.writeUTF(entry.getValue().toString());
      }
    }
  }

  @Override
  public String toString() {
    Vector vector = getVector();
    StringBuilder bldr = new StringBuilder("wt: ").append(getWeight()).append(' ');//打印该点到中心点的打分,即概率
    //打印附加的属性信息,比如该点到中心点的真实距离
    if (properties != null && !properties.isEmpty()) {
      for (Map.Entry<Text, Text> entry : properties.entrySet()) {
        bldr.append(entry.getKey().toString()).append(": ").append(entry.getValue().toString()).append(' ');
      }
    }
    //打印该向量点的具体内容
    bldr.append(" vec: ").append(vector == null ? "null" : AbstractCluster.formatVector(vector, null));
    return bldr.toString();
  }


}

