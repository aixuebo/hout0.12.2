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

package org.apache.mahout.common.parameters;

import org.apache.hadoop.conf.Configuration;

//说明要获取的参数是一个class类型的对象
public class ClassParameter extends AbstractParameter<Class> {
  
	/**
	 * @param prefix
	 * @param name 通过prefix.name获取class的实现类
	 * @param jobConf
	 * @param defaultValue 默认值就是class类型
	 * @param description
	 */
  public ClassParameter(String prefix, String name, Configuration jobConf, Class<?> defaultValue, String description) {
    super(Class.class, prefix, name, jobConf, defaultValue, description);
  }
  
  //将对应的value转换成class对象
  @Override
  public void setStringValue(String stringValue) {
    try {
      set(Class.forName(stringValue));
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
  }
  
  //返回class的全路径
  @Override
  public String getStringValue() {
    if (get() == null) {
      return null;
    }
    return get().getName();
  }
}
