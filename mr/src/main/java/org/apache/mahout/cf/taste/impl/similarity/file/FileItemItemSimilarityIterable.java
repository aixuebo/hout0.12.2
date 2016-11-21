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

package org.apache.mahout.cf.taste.impl.similarity.file;

import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

/**
 * {@link Iterable} to be able to read a file linewise into a {@link GenericItemSimilarity}
 * 从文件里面读取数据,每一个数据是item和item之间的相似度对象,即item1--item2--value
 */
final class FileItemItemSimilarityIterable implements Iterable<GenericItemSimilarity.ItemItemSimilarity> {

  private final File similaritiesFile;//相似文件

  FileItemItemSimilarityIterable(File similaritiesFile) {
    this.similaritiesFile = similaritiesFile;
  }

  //生成迭代器对象
  @Override
  public Iterator<GenericItemSimilarity.ItemItemSimilarity> iterator() {
    try {
      return new FileItemItemSimilarityIterator(similaritiesFile);
    } catch (IOException ioe) {
      throw new IllegalStateException("Can't read " + similaritiesFile, ioe);
    }
  }

}
