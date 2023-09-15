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

import elki.clustering.em.models.MultivariateGaussianModel;
import elki.clustering.hierarchical.linkage.CentroidLinkage;
import elki.clustering.hierarchical.linkage.Linkage;
import elki.clustering.hierarchical.linkage.SingleLinkage;
import elki.clustering.hierarchical.linkage.WardLinkage;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.database.ids.*;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.distance.minkowski.SquaredEuclideanDistance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.logging.statistics.DoubleStatistic;
import elki.math.linearalgebra.CovarianceMatrix;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import static elki.math.linearalgebra.VMath.*;

import elki.Algorithm;
import net.jafama.FastMath;

public class AWE<O, M extends MeanModel> extends AGNES<O> {
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(AGNES.class);
  
  
  /**
   * 
   * Constructor.
   *
   * @param distance Distance function to use
   * @param linkage Linkage method
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
   * Main worker instance of AWE.
   * 
   */
  public static class Instance<O extends NumberVector, M extends MeanModel> extends AGNES.Instance{
    /**
     * realtion
     */
    protected Relation<O> relation;
    /**
     * array to describe clusters that have ids of data which are in the cluster
     */
    protected ModifiableDBIDs[] clusters;
    /**
     * number of clusters
     */
    protected int r;
    /**
     * AWE values
     */
    protected double[] AWE;
    
    /**
     * Minimum loglikelihood to avoid -infinity.
     */
    protected static final double MIN_LOGLIKELIHOOD = -100000;

    /**
     * degree of freedom
     */
    protected int DEGREE_OF_FREEDOM = 0;
    /**
     * Key for statistics logging.
     */
    private static final String KEY = AWE.class.getName();
    
    /**
     * Constructor.
     *
     * @param linkage Linkage
     * @param relation relation
     */
    public Instance(Linkage linkage, Relation<O> relation) {
      super(linkage);
      this.relation = relation;
      this.clusters = new ModifiableDBIDs[relation.size()];
      this.r = 0;
      this.AWE = new double[relation.size()];
      AWE[0] = 0.;
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

      // print AWE values
      printAwe(AWE);

      return builder.complete();
    }
    /**
     * print a AWE array
     * 
     * @param awe awe array
     */
    private void printAwe(double[] awe){
      // shows upto AWE_25
      for(int i=0; i<25; i++){ 
        System.out.print(awe[i]+", ");
        LOG.statistics(new DoubleStatistic(KEY + ".AWE", awe[i]));
      }
      System.out.println("");
    }
    
    /**
     * initialize that clusters have the IDs of data
     * 
     * @param relation relation
     * @param clusterIds each clusterIds has the IDs of data that belong to a cluster
     */
    private void initClusterIds(Relation<O> relation, ModifiableDBIDs[] clusterIds) {
      int i=0;
      for (DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
        ModifiableDBIDs ids = DBIDUtil.newArray();
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
    }
    
    /**
     * update the array clusters and calculate the AWE values
     * 
     * @param clusters array for cluster
     * @param x First matrix position
     * @param y Second matrix position
     */
    private void updateClusters(ModifiableDBIDs[] clusters, int x, int y) {
      // p dimensional multivariate normal case
      final int dim = RelationUtil.dimensionality(relation);

      if(!clusters[x].isEmpty() && !clusters[y].isEmpty()) {
        ModifiableDBIDs cluster1 = clusters[x];
        ModifiableDBIDs cluster2 = clusters[y];
        ModifiableDBIDs mergedCluster = DBIDUtil.union(cluster1, cluster2);
        //lambda is the likelihood ratio test statistic
        double lambda = likelihoodRatioTestStatistic(cluster1, cluster2, mergedCluster);
        if(r>0){
          if(isSingleton(cluster1) && isSingleton(cluster2)) DEGREE_OF_FREEDOM= -dim; 
          else if(isSingleton(cluster1) || isSingleton(cluster2)) DEGREE_OF_FREEDOM = 0; // for the criterion Ward
          else DEGREE_OF_FREEDOM = dim;
          
          AWE[r] = lambda - (1.5 + FastMath.log(dim * mergedCluster.size())) * 2*DEGREE_OF_FREEDOM;
        }
        clusters[y].addDBIDs(clusters[x]);
        clusters[x].clear();
      }
    }
    
    /**
     * compute the Likelihood ratio test statistic (lambda_r)
     * 
     * @param cluster1 first cluster
     * @param cluster2 second cluster
     * @param mergedCluster merged cluster
     * @return Likelihood ratio test statistic
     */
    private double likelihoodRatioTestStatistic(ModifiableDBIDs cluster1, ModifiableDBIDs cluster2, ModifiableDBIDs mergedCluster) {
      //check if clusters[x] is a singleton and cluster[y] is singleton
      if(isSingleton(cluster1) && isSingleton(cluster2)) {
        r++;
        return 2.*logLikelihood(mergedCluster);
      }
      else if(isSingleton(cluster1) || isSingleton(cluster2)) {
        return isSingleton(cluster1) ? -2.*(logLikelihood(cluster2)-logLikelihood(mergedCluster)) : -2.*(logLikelihood(cluster1)-logLikelihood(mergedCluster));
      }
      else {
        r--;
        return 2*(logLikelihood(cluster1)+logLikelihood(cluster2) - logLikelihood(mergedCluster));
      }
    }
    /**
     * check @param cluster is singleton
     * 
     * @param cluster cluster
     * @return true if @param cluster is singleton
     */
    private boolean isSingleton(ModifiableDBIDs cluster){
      return cluster.size() < 2;
    }
    /**
     * compute the log-likelihood for @param cluster 
     * 
     * @param cluster cluster
     * @return log-likelihood
     */
    private double logLikelihood(ModifiableDBIDs cluster) {
      // create a model
      CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster);
      double[][] mat = cov.makePopulationMatrix();
      double[] means = cov.getMeanVector();
      // generate a multivariate Gaussian model to compute the log-likelihood
      MultivariateGaussianModel model = new MultivariateGaussianModel(1./r, means, mat);
      double[] logProbs = new double[cluster.size()];
      int i = 0;
      for(DBIDIter iditer = cluster.iter(); iditer.valid(); iditer.advance()){
        O vec = relation.get(iditer);
        logProbs[i++] = model.estimateLogDensity(vec);
      }
      return sum(logProbs);
    }
  }
  
  /**
   * Parameterization class
   */
  public static class Par<O, M extends MeanModel> extends AGNES.Par<O> {
    /**
     * Parameter to specify the EM cluster models to use.
     */
    public static final OptionID MODEL_ID = new OptionID("em.model", "Model factory.");
    /**
     * Parameter to specify the termination criterion for maximization of E(M):
     * E(M) - E(M') &lt; em.delta, must be a double equal to or greater than 0.
     */
    public static final OptionID DELTA_ID = new OptionID("em.delta", //
        "The termination criterion for maximization of E(M): E(M) - E(M') < em.delta");
    
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
    public AWE<O, M> make() {
      return new AWE<>(distance, linkage);
    }
  }

}
