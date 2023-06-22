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

import elki.clustering.em.EM;
import elki.clustering.em.models.MultivariateGaussianModel;
import elki.clustering.hierarchical.linkage.Linkage;
import elki.clustering.hierarchical.linkage.SingleLinkage;
import elki.data.NumberVector;
import elki.database.ids.*;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.math.linearalgebra.CovarianceMatrix;
import elki.utilities.optionhandling.Parameterizer;

import net.jafama.FastMath;

import static elki.math.linearalgebra.VMath.*;

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
  public static class Instance<O,V extends NumberVector> extends AGNES.Instance{
    /**
     * realtion
     */
    protected Relation<V> relation;
    /**
     * array to describe clusters that have ids of data which are in the cluster
     */
    protected ArrayModifiableDBIDs[] clusters;
    /**
     * number of clusters
     */
    protected int r;
    /**
     * AWE value
     * AWE[0] has the AWE value when the number of clusters is 1
     * AWE[148] has the AWE when the number of clusters is 149
     * the range of the number of clusters is between 1 and n-1, where n is the length of data.  
     */
    protected double[] AWE;
    /**
     * index for lambda 
     */
    protected int idx;
    
    /**
     * Minimum loglikelihood to avoid -infinity.
     */
    protected static final double MIN_LOGLIKELIHOOD = -100000;
    
    /**
     * Constructor.
     *
     * @param linkage Linkage
     */
    public Instance(Linkage linkage, Relation<V> relation) {
      super(linkage);
      this.relation = relation;
      this.clusters = new ArrayModifiableDBIDs[relation.size()];
      this.r = relation.size();
      this.AWE = new double[relation.size()-1];
      this.idx = AWE.length - 1;
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
    private void initClusterIds(Relation<V> relation, ArrayModifiableDBIDs[] clusterIds) {
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
     * update the array clusters and calculate the AWE values
     * 
     * @param clusters array for cluster
     * @param x First matrix position
     * @param y Second matrix position
     */
    private void updateClusters(ArrayModifiableDBIDs[] clusters, int x, int y) {
      if(!clusters[x].isEmpty() && !clusters[y].isEmpty()) {
        ArrayModifiableDBIDs cluster1 = clusters[x];
        ArrayModifiableDBIDs cluster2 = clusters[y];
        ModifiableDBIDs mergedCluster = DBIDUtil.union(cluster1, cluster2);
        //lambda is the likelihood ratio test statistic
        double lambda = likelihoodRatioTestStatistic(cluster1, cluster2, mergedCluster);
        // number of observations in the merged cluster
        int n = mergedCluster.size(); 
        // TODO number of degrees of freedom in asymptotic chi square of lambda (between1-5)
        int delta = numberOfFreeParameters();
        // TODO p-dimensional multivariate normal case : 클러스터 갯수? 왜냐면 클러스터 하나당 하나의 MVN분포를 갖기때문에
        int p = 1;
        // compute AWE
        AWE[idx--] = lambda - 2*delta*(1.5 + FastMath.log(p*n));
        clusters[y].addDBIDs(clusters[x]);
        clusters[x].clear();
      }
    }
    
    /**
     * compute the Likelihood ratio test statistic (lambda_r)
     * @param cluster1 first cluster
     * @param cluster2 second cluster
     * @param mergedCluster merged cluster
     * @return Likelihood ratio test statistic
     */
    private double likelihoodRatioTestStatistic(ArrayModifiableDBIDs cluster1, ArrayModifiableDBIDs cluster2, ModifiableDBIDs mergedCluster) {
      //check if clusters[x] is a singleton and cluster[y] is singleton
      if(cluster1.size()==1 && cluster2.size()==1) {
        return 2.*loglikelihood(mergedCluster);
      }
      else if(cluster1.size()==1 || cluster2.size()==1) {
        return cluster1.size()==1 ? 2.*(loglikelihood(cluster2)-loglikelihood(mergedCluster)) : 2.*(loglikelihood(cluster1)-loglikelihood(mergedCluster));
      }
      else {
        return 2*(loglikelihood(cluster1)+loglikelihood(cluster2) - loglikelihood(mergedCluster));
      }
    }
    
    /**
     * compute loglikelihood for normal cluster
     * 
     * @param cluster1
     * @return loglikelihood
     */
    // TODO 계산이 이상함
    private double loglikelihood(ArrayModifiableDBIDs cluster) {
      //create a model
      CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster);
      double[][] mat = cov.destroyToSampleMatrix();
      double[] means = cov.getMeanVector();
      int k = means.length;
      MultivariateGaussianModel model = new MultivariateGaussianModel(1. / k, means, mat);
      
      //calculate the log likelihood
      double[] probs = new double[cluster.size()];
      int i=0;
      for(DBIDIter iditer = cluster.iter(); iditer.valid(); iditer.advance()) {
        V vec = relation.get(iditer);
        double v = model.estimateLogDensity(vec);
        probs[i++] = v > MIN_LOGLIKELIHOOD ? v : MIN_LOGLIKELIHOOD;
      }
      //loglikelihood = logSumExp( log p(x) )
      return EM.logSumExp(probs);
    }
    /**
     * for merged cluster
     * @param cluster
     * @return loglikelihood
     */
    private double loglikelihood(ModifiableDBIDs cluster) {
      //create a model
      CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster);
      double[][] mat = cov.destroyToSampleMatrix();
      double[] means = cov.getMeanVector();
      int k = means.length;
      MultivariateGaussianModel model = new MultivariateGaussianModel(1. / k, means, mat);
      
      //calculate the log likelihood
      double[] probs = new double[cluster.size()];
      int i=0;
      for(DBIDIter iditer = cluster.iter(); iditer.valid(); iditer.advance()) {
        V vec = relation.get(iditer);
        double v = model.estimateLogDensity(vec);
        probs[i++] = v > MIN_LOGLIKELIHOOD ? v : MIN_LOGLIKELIHOOD;
      }
      //loglikelihood = LogSumExp( log p(x) )
      return EM.logSumExp(probs);
    }
    /** TODO
     * Compute the number of free parameters.
     *
     * @param k number of clusters
     * @param mergedCluster the merged cluster
     * @return Number of free parameters
     */
    private int numberOfFreeParameters() {
      
      return 0;
    }
  }
  
  public static class Par<O> extends AGNES.Par<O> implements Parameterizer {
    
    @Override
    public AWE<O> make() {
      return new AWE<>(distance, linkage);
    }
  }

}
