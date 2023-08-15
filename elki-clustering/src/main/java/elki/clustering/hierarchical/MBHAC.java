/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2022
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
import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.ArrayDBIDs;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDUtil;
import elki.database.ids.ModifiableDBIDs;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.math.MathUtil;
import elki.math.linearalgebra.CovarianceMatrix;
import elki.math.linearalgebra.EigenvalueDecomposition;
import elki.utilities.exceptions.AbortException;
import elki.utilities.optionhandling.Parameterizer;
import net.jafama.FastMath;

import static elki.math.linearalgebra.VMath.*;

/**
 * Model Based Gaussian Hierarchical Clustering
 */
public class MBHAC<O extends NumberVector> implements HierarchicalClusteringAlgorithm {
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(MBHAC.class);

  /**
   * array to describe clusters that have ids of data which are in the cluster
   */
  protected ModifiableDBIDs[] clusters;
  /**
   * number of clusters
   */
  protected int r; //
  /**
   * AWE value
   */
  protected double[] AWE;

  /**
   * degree of freedom for the criterion S* in the paper "model based Gaussian and Non Gaussian Clustering (1993) by Jeffrey D.Banfield and Adrian E. Raftery"
   */
  protected final int DEGREE_OF_FREEDOM = 4; // for the criterion S*

  /**
   * Constructor.
   * 
   */
  public MBHAC() {
  }
  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
  }
  /**
   * init that clusters have the ids of data
   * 
   * @param relation relation
   * @param clusterIds is a array for DBIDs of the data in each cluster
   */
  private void initClusterIds(Relation<O> relation, ModifiableDBIDs[] clusterIds) {
    int i=0;
    for (DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
      ModifiableDBIDs ids = DBIDUtil.newArray();
      ids.add(iditer);
      clusterIds[i++] = ids;
    }
  }

  /**
   * compute the criterion in the paper [by Banfield and Raftery] for S*
   * This function should be called after the two clusters have been merged (@param clusters is the state after the merging).
   * 
   * @param relation relation
   * @param clusters is a array for DBIDs of the data in each cluster
   * @return merging cost
   */
  private double criterion(Relation<O> relation, ModifiableDBIDs[] clusters){
    double sum_S = 0.;
    for(int i=0; i<clusters.length; i++){
      sum_S += criterionForOneCluster(relation, clusters[i]);
    }
    return sum_S;
  }
  /**
   * compute n_k log (S_k / n_k) s.t. S_k = tr(A⁻1 * Omega_k)
   * 
   * @param relation relation
   * @param cluster is a array for DBIDs of the data in each cluster
   * @return result of calculating for n_k log (S_k/n_k)
   */
  private double criterionForOneCluster(Relation<O> relation, ModifiableDBIDs cluster){
    // if cluster is the cluster that is aleady merged. (except for singleton)
    if(cluster.size() < 2) return 0.;

    int n = cluster.size();
    CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster);
    double[][] mat = cov.makePopulationMatrix();
    double[] means = cov.getMeanVector();

    EigenvalueDecomposition evd = new EigenvalueDecomposition(mat);
    double[] diag = getDiagonal(evd.getD()); // diag is sorted by descending
    double lambda_k = diag[0]; // first eigenvalue
    double[] A = times(diag, 1./lambda_k); 
    double[] inv_A = new double[A.length]; // inverse of A
    for(int i=0; i<A.length; i++){
      if(A[i] == 0) {
        inv_A[i] = 0.;
      }
      else{
        inv_A[i] = 1./A[i];
      }
    }

    // compute W = sum[(x - mu)(x - mu)^T] (W/n = MLE of covariance)
    int dim = RelationUtil.dimensionality(relation);
    double[][] W = new double[dim][dim];
    for(DBIDIter dbiter = cluster.iter(); dbiter.valid(); dbiter.advance()){
      O vec = relation.get(dbiter);
      double[] xi = vec.toArray();
      W = plus(W, timesTranspose(minus(xi, means), minus(xi, means)));
    }

    //eigenvalue of W using eigenvalue decomposition
    EigenvalueDecomposition evd_W = new EigenvalueDecomposition(W);
    double[][] omega = evd_W.getD();

    // S_k = tr(A⁻1 * eigenvalue of W)
    double S = sum(getDiagonal(times(diagonal(inv_A), omega)));
    
    // n_k log (S_k/n_k)
    return n * FastMath.log(S / n);
  }
  /**
   * merge two clusters
   * 
   * @param relation relation
   * @param clusters is a array for DBIDs of the data in each cluster
   * @param cidx cluster ids : Starting clusters are assigned ID values from 1 to n. If two clusters are merged, the cluster that has no data is assigned -1 and the merged cluster has a new ID value.
   */
  private void merge(Relation<O> relation, ModifiableDBIDs[] clusters, int[] cidx, ClusterMergeHistoryBuilder builder,double[] scratch){
    int size = clusters.length; // == relation.size()

    for(int i=1; i<size; i++){
      final int ibase = triangleSize(i);
      for(int j=0; j<i; j++){
        if(clusters[i].size() < 1 || clusters[j].size() < 1){
          scratch[ibase + j] = Double.POSITIVE_INFINITY;
          continue;
        }
        // deep copy
        ModifiableDBIDs[] copy = new ModifiableDBIDs[clusters.length];
        for(int h=0;h<copy.length; h++){
          copy[h] = DBIDUtil.newArray(clusters[h]);
        }
        
        // merge i->j 
        copy[j].addDBIDs(copy[i]);
        copy[i].clear();
        // compute criterion S* 
        double S_star = criterion(relation, copy);
        scratch[ibase + j] = S_star;
      }
    }

    // to find the minimun in scratch
    double min = Double.POSITIVE_INFINITY;
    int minx = -1, miny = -1;
    for(int x = 1; x < size; x++) {
      if(cidx[x] >= 0) {
        final int xbase = triangleSize(x);
        for(int y = 0; y < x; y++) {
          if(cidx[y] >= 0) {
            final int idx = xbase + y;
            if(scratch[idx] <= min) {
              min = scratch[idx];
              minx = x;
              miny = y;
            }
          }
        }
      }
    }
    assert minx >= 0 && miny >= 0;
    scratch = null;
    // merge
    //update clusters ids (+ merging minx, miny clusters)
    updateClusters(relation, clusters, minx, miny);

    int zz = builder.add(minx, min, miny);
    cidx[minx] = -1; // the cluster that has no data is assigned -1
    cidx[miny] = zz; // the merged new cluster has a new ID value.
  }

  private void printArr(double[] arr){
    for(int i=0; i<arr.length; i++){
      System.out.print(arr[i]+", ");
    }
    System.out.println("");
  }

  /**
   * Run the algorithm
   *
   * @param relation Relation
   * @return Clustering hierarchy
   */
  public ClusterMergeHistory run(Relation<O> relation) {
    
    ArrayDBIDs ids = DBIDUtil.ensureArray(relation.getDBIDs());
    final int size = ids.size();

    if(size > 0x10000) {
      throw new AbortException("This implementation does not scale to data sets larger than " + 0x10000 + " instances (~17 GB RAM), which results in an integer overflow.");
    }

    // init for clusters ids
    this.clusters = new ModifiableDBIDs[size];
    initClusterIds(relation, clusters);

    // r+1 when two singletons are merged, r is unchanged when a sigleton and a cluster are merged, and r-1 when two clusters are merged.
    this.r = 0;
    this.AWE = new double[size];
    AWE[0] = 0.;

    // 즉 (recursive로) 각각의 두점씩 Merge한 상태의 S*값은 구하고 scratch에서 최소값을 가진 두 클러스터의 merge를 남기고 반복. (위의 merge함수에 구현)
    // get each of the two clusters, merge them, calculate S*, and save the result in a scratch array
    double[] scratch = new double[triangleSize(size)]; // merging cost를 저장하는 곳
    
    ClusterMergeHistoryBuilder builder = new ClusterMergeHistoryBuilder(ids, false);

    int[] cidx = MathUtil.sequence(0, size);
    // Repeat until everything merged, except the desired number of clusters:
    FiniteProgress prog = LOG.isVerbose() ? new FiniteProgress("Agglomerative clustering", size - 1, LOG) : null;
    for(int i = 1; i < size; i++) {
      System.out.println(i + ". merging");
      merge(relation, clusters, cidx, builder, scratch);
      LOG.incrementProcessed(prog);
      System.out.println(i + ". merging is done");
    }
    LOG.ensureCompleted(prog);
    printArr(AWE);
    return builder.complete();
  }
 
  /**
   * Compute the size of a complete x by x triangle (minus diagonal)
   * 
   * @param x Offset
   * @return Size of complete triangle
   */
  protected static int triangleSize(int x) {
    return (x * (x - 1)) >>> 1;
  }
  
  /**
   * update the array clusters and calculate the AWE values
   * 
   * @param clusters array for cluster
   * @param x First matrix position
   * @param y Second matrix position
   */
  private void updateClusters(Relation<O> relation, ModifiableDBIDs[] clusters, int x, int y) {
    // p dimensional multivariate normal case
    final int p = RelationUtil.dimensionality(relation);

    if(!clusters[x].isEmpty() && !clusters[y].isEmpty()) {
      ModifiableDBIDs cluster1 = clusters[x];
      ModifiableDBIDs cluster2 = clusters[y];
      ModifiableDBIDs mergedCluster = DBIDUtil.union(cluster1, cluster2);
      //lambda is the likelihood ratio test statistic
      double lambda = likelihoodRatioTestStatistic(relation, cluster1, cluster2, mergedCluster);
      if(r>0){
        // AWE_r = lambda - (1.5 + log(p * n))* 2* degree of freedom // 
        AWE[r] += lambda - (1.5 + FastMath.log(p * mergedCluster.size())) * 2*DEGREE_OF_FREEDOM;
      }
      else {
        System.out.println("Something wrong");
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
  private double likelihoodRatioTestStatistic(Relation<O> relation, ModifiableDBIDs cluster1, ModifiableDBIDs cluster2, ModifiableDBIDs mergedCluster) {
    //check if cluster1 is a singleton and cluster2 is singleton
    if(isSingleton(cluster1) && isSingleton(cluster2)) {
      r++;
      return 2.*logLikelihood(relation, mergedCluster);
    }
    else if(isSingleton(cluster1) || isSingleton(cluster2)) {
      return isSingleton(cluster1) ? -2.*(logLikelihood(relation, cluster2)-logLikelihood(relation, mergedCluster)) : -2.*(logLikelihood(relation, cluster1)-logLikelihood(relation, mergedCluster));
    }
    else {
      r--;
      return 2*(logLikelihood(relation, cluster1)+logLikelihood(relation, cluster2) - logLikelihood(relation, mergedCluster));
    }
  }
  private boolean isSingleton(ModifiableDBIDs cluster){
    return cluster.size() < 2;
  }
    
  private double logLikelihood(Relation<O> relation, ModifiableDBIDs cluster) {
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

  /**
   * Parameterization class
   * 
   * @param <O> Object type
   */
  public static class Par<O> implements Parameterizer {

    @Override
    public MBHAC make() {
      return new MBHAC();
    }
  }
}
