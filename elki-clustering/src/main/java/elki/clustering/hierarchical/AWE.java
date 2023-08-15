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

import elki.clustering.em.models.EMClusterModelFactory;
import elki.clustering.em.models.MultivariateGaussianModel;
import elki.clustering.em.models.MultivariateGaussianModelFactory;
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
import elki.math.linearalgebra.CovarianceMatrix;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import static elki.math.linearalgebra.VMath.*;

import elki.Algorithm;
import net.jafama.FastMath;

public class AWE<O, M extends MeanModel> extends AGNES<O> {
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(AGNES.class);
  
  protected EMClusterModelFactory<? super O, M> mfactory;
  protected double delta;
  
  /**
   * 
   * Constructor.
   *
   * @param distance
   * @param linkage
   */
  public AWE(Distance<? super O> distance, Linkage linkage, EMClusterModelFactory<? super O, M> mfactory, double delta) {
    super(distance, linkage);
    this.mfactory = mfactory;
    this.delta = delta;
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
    return new Instance(super.linkage, relation, mfactory, delta).run(mat, new ClusterMergeHistoryBuilder(ids, super.distance.isSquared()));
  }
  /**
   * Main worker instance of AGNES.
   * 
   * @author Erich Schubert
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
     * AWE value
     */
    protected double[] AWE;
    
    /**
     * Minimum loglikelihood to avoid -infinity.
     */
    protected static final double MIN_LOGLIKELIHOOD = -100000;
    
    protected EMClusterModelFactory<? super O, M> mfactory;
    protected double delta;
    /**
     * degree of freedom for the criterion S* in the paper "model based Gaussian and Non Gaussian Clustering (1993) by Jeffrey D.Banfield and Adrian E. Raftery"
     */
    protected final int DEGREE_OF_FREEDOM = 2; // for the criterion Ward
    
    /**
     * Constructor.
     *
     * @param linkage Linkage
     */
    public Instance(Linkage linkage, Relation<O> relation, EMClusterModelFactory<? super O, M> mfactory, double delta) {
      super(linkage);
      this.relation = relation;
      this.clusters = new ModifiableDBIDs[relation.size()];
      this.r = 0;
      this.AWE = new double[relation.size()];
      AWE[0] = 0.;
      this.mfactory = mfactory;
      this.delta = delta;
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
      printArr(AWE);
      return builder.complete();
    }
    private void printArr(double[] arr){
      for(int i=0; i<arr.length; i++){
        System.out.print(arr[i]+", ");
      }
      System.out.println("");
    }
    
    /**
     * init that clusters have the ids of data
     * 
     * @param relation
     * @param clusterIds
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
      final int p = RelationUtil.dimensionality(relation);

      if(!clusters[x].isEmpty() && !clusters[y].isEmpty()) {
        ModifiableDBIDs cluster1 = clusters[x];
        ModifiableDBIDs cluster2 = clusters[y];
        ModifiableDBIDs mergedCluster = DBIDUtil.union(cluster1, cluster2);
        //lambda is the likelihood ratio test statistic
        double lambda = likelihoodRatioTestStatistic(cluster1, cluster2, mergedCluster);
        if(r>0){
          AWE[r] += lambda - (1.5 + FastMath.log(p * mergedCluster.size())) * 2*DEGREE_OF_FREEDOM;;
        }
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
    private boolean isSingleton(ModifiableDBIDs cluster){
      return cluster.size() < 2;
    }
    
    private double logLikelihood(ModifiableDBIDs cluster) {
      // create a model
      CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster);
      double[][] mat = cov.makePopulationMatrix();
      double[] means = cov.getMeanVector();
      MultivariateGaussianModel model = new MultivariateGaussianModel(1./means.length, means, mat);
      double[] logProbs = new double[cluster.size()];
      int i = 0;
      for(DBIDIter iditer = cluster.iter(); iditer.valid(); iditer.advance()){
        O vec = relation.get(iditer);
        logProbs[i++] = model.estimateLogDensity(vec);
      }
      return sum(logProbs);
    }
  }
  
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

    /**
     * Cluster model factory.
     */
    protected EMClusterModelFactory<O, M> mfactory;
    /**
     * Stopping threshold
     */
    protected double delta;
    
    @Override
    public void configure(Parameterization config) {
      new ObjectParameter<EMClusterModelFactory<O, M>>(MODEL_ID, EMClusterModelFactory.class, MultivariateGaussianModelFactory.class) //
          .grab(config, x -> mfactory = x);
      new DoubleParameter(DELTA_ID, 1e-7)// 
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE) //
          .grab(config, x -> delta = x);
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
      return new AWE<>(distance, linkage, mfactory, delta);
    }
  }

}
