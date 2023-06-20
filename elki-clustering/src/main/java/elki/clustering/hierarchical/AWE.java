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

import elki.clustering.hierarchical.linkage.CentroidLinkage;
import elki.clustering.hierarchical.linkage.Linkage;
import elki.clustering.hierarchical.linkage.SingleLinkage;
import elki.clustering.hierarchical.linkage.WardLinkage;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.ArrayDBIDs;
import elki.database.ids.DBIDUtil;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.distance.minkowski.SquaredEuclideanDistance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.Algorithm;
import elki.clustering.hierarchical.AGNES.Instance;

public class AWE<O> implements HierarchicalClusteringAlgorithm{
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
   * Agglomerative Nesting
   */
  protected AGNES agnes;
  
  /**
   * 
   * Constructor.
   *
   * @param distance
   * @param linkage
   */
  public AWE(Distance<? super O> distance, Linkage linkage) {
    this.distance = distance;
    this.linkage = linkage;
    this.agnes = new AGNES(distance, linkage);
  }
  /**
   * run the main algorithm
   * 
   * @param relation
   * @return result
   */
  public ClusterMergeHistory run(Relation<O> relation) {
    if(SingleLinkage.class.isInstance(linkage)) {
      LOG.verbose("Notice: SLINK is a much faster algorithm for single-linkage clustering!");
    }
    final ArrayDBIDs ids = DBIDUtil.ensureArray(relation.getDBIDs());
    // Compute the initial (lower triangular) distance matrix.
    DistanceQuery<O> dq = new QueryBuilder<>(relation, distance).distanceQuery();
    
    Instance ins = new Instance(linkage);
    ins.builder = new ClusterMergeHistoryBuilder(ids, distance.isSquared());
    ins.mat = agnes.initializeDistanceMatrix(ids, dq, linkage);
    
    final int size = ins.mat.size;
    ins.end = size;
    // Repeat until everything merged into 1 cluster
    FiniteProgress prog = LOG.isVerbose() ? new FiniteProgress("Agglomerative clustering", size - 1, LOG) : null;
    // Use end to shrink the matrix virtually as the tailing objects disappear
    for(int i = 1; i < size; i++) {
      ins.end = ins.shrinkActiveSet(ins.mat.clustermap, ins.end, ins.findMerge());
      LOG.incrementProcessed(prog);
    }
    LOG.ensureCompleted(prog);
    return ins.builder.complete();
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(distance.getInputTypeRestriction());
  }
  public static class Par<O> implements Parameterizer {
    /**
     * Option ID for linkage parameter.
     */
    public static final OptionID LINKAGE_ID = new OptionID("hierarchical.linkage", "Linkage method to use (e.g., Ward, Single-Link)");

    /**
     * Current linkage in use.
     */
    protected Linkage linkage;

    /**
     * The distance function to use.
     */
    protected Distance<? super O> distance;

    @Override
    public void configure(Parameterization config) {
      new ObjectParameter<Linkage>(LINKAGE_ID, Linkage.class) //
          .setDefaultValue(WardLinkage.class) //
          .grab(config, x -> linkage = x);
      Class<? extends Distance<?>> defaultD = (linkage instanceof WardLinkage || linkage instanceof CentroidLinkage) //
          ? SquaredEuclideanDistance.class : EuclideanDistance.class;
      new ObjectParameter<Distance<? super O>>(Algorithm.Utils.DISTANCE_FUNCTION_ID, Distance.class, defaultD) //
          .grab(config, x -> distance = x);
    }

    @Override
    public AWE<O> make() {
      return new AWE<>(distance, linkage);
    }
  }
}
