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
import elki.clustering.hierarchical.linkage.SingleLinkage;
import elki.database.ids.*;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.utilities.optionhandling.Parameterizer;

public class AWE<O> extends AGNES<O> implements HierarchicalClusteringAlgorithm{
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(AGNES.class);
  
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
  @Override
  public ClusterMergeHistory run(Relation<O> relation) {
    if(SingleLinkage.class.isInstance(super.linkage)) {
      LOG.verbose("Notice: SLINK is a much faster algorithm for single-linkage clustering!");
    }
    final ArrayDBIDs ids = DBIDUtil.ensureArray(relation.getDBIDs());
    // Compute the initial (lower triangular) distance matrix.
    DistanceQuery<O> dq = new QueryBuilder<>(relation, super.distance).distanceQuery();
    ClusterDistanceMatrix mat = AGNES.initializeDistanceMatrix(ids, dq, super.linkage);
    return new Instance(super.linkage, relation).run(mat, new ClusterMergeHistoryBuilder(ids, super.distance.isSquared()));
  }
  /**
   * Main worker instance of AGNES.
   * 
   * @author Erich Schubert
   */
  public static class Instance<O> extends AGNES.Instance{
    /**
     * realtion
     */
    protected Relation<O> relation;
    /**
     * array to describe clusters that have ids of data which are in the cluster
     */
    protected ArrayModifiableDBIDs[] clusters;
    /**
     * number of clusters
     */
    protected int r;
    /**
     * the likelihood ratio test statistic
     * 
     * lambda[r-1] has the value of lambda_r (r = {1,...,n-1}, n is the number of data)
     * lambda[0] has the value that is calculated when last two clusters(r=2) are merged to one cluster (r=1) 
     */
    protected double[] lambda;
    
    /**
     * Minimum loglikelihood to avoid -infinity.
     */
    protected static final double MIN_LOGLIKELIHOOD = -100000;
    
    /**
     * Constructor.
     *
     * @param linkage Linkage
     */
    public Instance(Linkage linkage, Relation<O> relation) {
      super(linkage);
      this.relation = relation;
      this.clusters = new ArrayModifiableDBIDs[relation.size()];
      this.r = relation.size();
      this.lambda = new double[relation.size()-1];
    }
    
    @Override
    public ClusterMergeHistory run(ClusterDistanceMatrix mat, ClusterMergeHistoryBuilder builder) {
      final int size = mat.size;
      super.mat = mat;
      super.builder = builder;
      super.end = size;
      initClusterIds(relation, clusters);
      // Repeat until everything merged into 1 cluster
      FiniteProgress prog = LOG.isVerbose() ? new FiniteProgress("Agglomerative clustering", size - 1, LOG) : null;
      // Use end to shrink the matrix virtually as the tailing objects disappear
      for(int i = 1; i < size; i++) {
        super.end = shrinkActiveSet(mat.clustermap, super.end, findMerge());
        LOG.incrementProcessed(prog);
      }
      LOG.ensureCompleted(prog);
      return builder.complete();
    }
    
    /**
     * init that clusters have the ids of data
     * 
     * @param relation
     * @param clusterIds
     */
    private void initClusterIds(Relation<O> relation, ArrayModifiableDBIDs[] clusterIds) {
      int i=0;
      for (DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
        ArrayModifiableDBIDs ids = DBIDUtil.newArray();
        ids.add(iditer);
        clusterIds[i++] = ids;
      }
    }
    
    @Override
    protected void merge(double mindist, int x, int y) {
      assert x >= 0 && y >= 0;
      assert y < x; // more efficient
      final int xx = super.mat.clustermap[x], yy = super.mat.clustermap[y];
      final int sizex = super.builder.getSize(xx), sizey = super.builder.getSize(yy);
      int zz = super.builder.strictAdd(xx, super.linkage.restore(mindist, super.builder.isSquared), yy);
      assert super.builder.getSize(zz) == sizex + sizey;
      // Since y < x, prefer keeping y, dropping x.
      super.mat.clustermap[y] = zz;
      super.mat.clustermap[x] = -1; // deactivate
      updateMatrix(mindist, x, y, sizex, sizey);
      updateClusters(clusters, x, y);
      r--;
    }
    
    /**
     * update the array clusters
     * 
     * @param clusters array for cluster
     * @param x First matrix position
     * @param y Second matrix position
     */
    private void updateClusters(ArrayModifiableDBIDs[] clusters, int x, int y) {
      if(!clusters[x].isEmpty() && !clusters[y].isEmpty()) {
        ArrayModifiableDBIDs cluster1 = clusters[x];
        ArrayModifiableDBIDs cluster2 = clusters[y];
        clusters[y].addDBIDs(clusters[x]);
        clusters[x].clear();
        ArrayModifiableDBIDs mergedCluster = clusters[y];
        calculateLikelihoodRatioTestStatistic(cluster1, cluster2, mergedCluster);
      }
    }
    
    /**
     * calculate the Likelihood ratio test statistic 
     * @param cluster1 first cluster
     * @param cluster2 second cluster
     * @param mergedCluster merged cluster
     */
    private void calculateLikelihoodRatioTestStatistic(ArrayModifiableDBIDs cluster1, ArrayModifiableDBIDs cluster2, ArrayModifiableDBIDs mergedCluster) {
      //TODO check if clusters[x] is a singleton and cluster[y] is singleton
      if(cluster1.size()==1 && cluster2.size()==1) {
        
      }
      else if(cluster1.size()==1 || cluster2.size()==1) {
        
      }
      else {
        
      }
      //TODO calculate the log likelihood of two clusters which will be merged
      //TODO calculate the log likelihood of the merged cluster above
      //TODO derive lambda_r and store into the lambda array
    }
    
    /**
     * calculate loglikelihood 
     * 
     * @param cluster1
     * @return loglikelihood
     */
    private double loglikelihood(ArrayModifiableDBIDs cluster) {
      for(DBIDIter iditer = cluster.iter(); iditer.valid(); iditer.advance()) {
        O vec = relation.get(iditer);
        //TODO data의 mean과 var을 계산후 normal distribution model을 만들어서 log likelihood계산하기
      }
      return 0.;
    }
  }
  
  public static class Par<O> extends AGNES.Par<O> implements Parameterizer {
    
    @Override
    public AWE<O> make() {
      return new AWE<>(distance, linkage);
    }
  }

}
