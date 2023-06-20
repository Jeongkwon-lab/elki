/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2023
 * ELKI Development Team
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.clustering.hierarchical;

import elki.clustering.hierarchical.linkage.Linkage;
import elki.clustering.hierarchical.linkage.WardLinkage;
import elki.distance.Distance;
import elki.logging.Logging;
import elki.utilities.optionhandling.Parameterizer;

public class AWE<O> extends AGNES implements HierarchicalClusteringAlgorithm{
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(AGNES.class);
  
  /**
   * Distance function used.
   */
  protected Distance<? super O> distance;

  /**
   * Current linkage method in use.
   */
  protected Linkage linkage = WardLinkage.STATIC;
  
  
  /**
   * 
   * Constructor.
   *
   * @param distance
   * @param linkage
   */
  public AWE(Distance<? super O> distance, Linkage linkage) {
    super(distance, linkage);
  }
  
  /**
   * Main worker instance of AGNES.
   * 
   * @author Erich Schubert
   */
  public static class Instance extends AGNES.Instance{
    /**
     * Current linkage method in use.
     */
    protected Linkage linkage;

    /**
     * Cluster distance matrix
     */
    protected ClusterDistanceMatrix mat;

    /**
     * Cluster result builder
     */
    protected ClusterMergeHistoryBuilder builder;

    /**
     * Active set size
     */
    protected int end;

    /**
     * Constructor.
     *
     * @param linkage Linkage
     */
    public Instance(Linkage linkage) {
      super(linkage);
    }
    //TODO merge함수에 n갯수만큼의 int[] clusterIds(인덱스 번호는 클러스터 번호, 내용은 데이터의 DBIDs를 담고있다.) = new int[k]를 만들어서 실제로 merge될때 clusterIds 두개의 합쳐지는 arr를 합치기.
    // mat.clustermap은 클러스터의 번호는 담고있다. 처음에는 0-149까지 있고 합쳐질때마다 150++ 다른하나는 -1의 값을 갖는다.
    @Override
    protected void merge(double mindist, int x, int y) {
      
    }
  }
  
  public static class Par<O> extends AGNES.Par<O> implements Parameterizer {
    
    @Override
    public AWE<O> make() {
      return new AWE<>(distance, linkage);
    }
  }
}
