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
package elki.clustering.em;


import static elki.math.linearalgebra.VMath.*;

import java.util.*;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.em.models.EMClusterModelFactory;
import elki.clustering.em.models.MultivariateGaussianModelFactory;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.math.linearalgebra.pca.PCAResult;
import elki.math.linearalgebra.pca.PCARunner;
import elki.math.linearalgebra.pca.StandardCovarianceMatrixBuilder;
import elki.math.statistics.tests.AndersonDarlingTest;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.*;
import elki.utilities.random.RandomFactory;

public class PGMeans_PCA_ADT<O extends NumberVector, M extends MeanModel> implements ClusteringAlgorithm<Clustering<M>>{
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(PGMeans_PCA_ADT.class);

  protected int k = 1;
  protected double delta;

  protected EMClusterModelFactory<? super O, M> mfactory;
  protected RandomFactory random;
  protected double critical;

  /**
   *
   * Constructor.
   *
   * @param delta delta parameter
   * @param mfactory EM cluster model factory
   * @param mprojection Random projection family
   * @param critical for AD-Test
   */
  public PGMeans_PCA_ADT(double delta, EMClusterModelFactory<? super O, M> mfactory, RandomFactory random, double critical){
    this.delta = delta;
    this.mfactory = mfactory;
    this.random = random;
    this.critical = critical;
  }
  /**
   * Performs the PG-Means algorithm on the given database.
   *
   * @param relation to use
   * @return result
   */
  public Clustering<M> run(Relation<O> relation) {
    if(relation.size() == 0) {
      throw new IllegalArgumentException("database empty: must contain elements");
    }

    // PG-Means
    boolean rejected = true;
    while(rejected) {
      EM<O, M> em = new EM<O, M>(k, delta, mfactory);
      Clustering<M> clustering = em.run(relation);
      rejected = testResult(relation, clustering);
      if(rejected) {
        k++;
      }
      // in general, the number of clusters is within 10
      if(k>100){
        System.out.println("KS-Test is going to be wrong");
        break;
      }
    }

    System.out.println("k :" + k);
    return new EM<O, M>(k, delta, mfactory).run(relation);
  }

  /**
   * generate a random projection,
   * and project the dataset and model,
   * Then, KS-test
   *
   * @param relation
   * @param clustering the result of em with k
   * @param p number of projections
   * @return true if the test is rejected
   */
  private boolean testResult(Relation<O> relation, Clustering<M> clustering) {
    boolean rejected = false;

    // pca depends on the variance of cluster. So we have not to repeat the test p-times
    ArrayList<Cluster<M>> clusters = new ArrayList<>(clustering.getAllClusters());

    for(Cluster<M> cluster : clusters) {
      double[] pcaFilter = runPCA(relation, cluster);
      double[][] data = new double[cluster.size()][RelationUtil.dimensionality(relation)];

      int j=0;
      // TODO data standardize?
      for(DBIDIter iditer = cluster.getIDs().iter(); iditer.valid(); iditer.advance()) {
        O vec = relation.get(iditer);
        data[j++] = vec.toArray();
      }

      double[][] test = transposeTimes(pcaFilter, transpose(data));
      double[] projectedData = test[0];

      // AD-Test to pca projected data
      Arrays.sort(projectedData);
      double A2 = AndersonDarlingTest.A2Noncentral(projectedData);
      A2 = AndersonDarlingTest.removeBiasNormalDistribution(A2, projectedData.length);
      if(A2 > critical) {
        //rejected
        return rejected = true;
      }
    }
    return rejected;
  }
  /**
   * run PCA using ELKI
   *
   * @param relation
   * @return filter that outputs one dimension matrix
   */
  private double[] runPCA(Relation<O> relation, Cluster<M> cluster) {
    StandardCovarianceMatrixBuilder scov = new StandardCovarianceMatrixBuilder();
    PCARunner pca = new PCARunner(scov);
    PCAResult pcaResult = pca.processIds(cluster.getIDs(), relation);

    // return first Vector in sorted EigenPairs, because the filter outputs one dimension
    double[][] eigenvectors = pcaResult.getEigenvectors();
    double[] filter = new double[eigenvectors.length];
    for(int i=0; i<filter.length; i++) {
      filter[i] = eigenvectors[i][0];
    }
    return filter;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(mfactory.getInputTypeRestriction());
  }

  public static class Par<O, M extends MeanModel> implements Parameterizer {

    /**
     * Parameter to specify the termination criterion for maximization of E(M):
     * E(M) - E(M') &lt; em.delta, must be a double equal to or greater than 0.
     */
    public static final OptionID DELTA_ID = new OptionID("em.delta", //
        "The termination criterion for maximization of E(M): E(M) - E(M') < em.delta");

    /**
     * Parameter to specify the EM cluster models to use.
     */
    public static final OptionID MODEL_ID = new OptionID("em.model", "Model factory.");

    /**
     * Parameter to specify a minimum number of iterations.
     */
    public static final OptionID MINITER_ID = new OptionID("em.miniter", "Minimum number of iterations.");

    /**
     * Parameter to specify the maximum number of iterations.
     */
    public static final OptionID MAXITER_ID = new OptionID("em.maxiter", "Maximum number of iterations.");

    /**
     * Parameter to specify the MAP prior
     */
    public static final OptionID PRIOR_ID = new OptionID("em.map.prior", "Regularization factor for MAP estimation.");

    /**
     * Parameter to specify the saving of soft assignments
     */
    public static final OptionID SOFT_ID = new OptionID("em.soft", "Retain soft assignment of clusters.");

    /**
     * Randomization seed.
     */
    public static final OptionID SEED_ID = new OptionID("pgmeans.seed", "Random seed for splitting clusters.");
    /**
     * Critical value for the Anderson-Darling-Test
     */
    public static final OptionID CRITICAL_ID = new OptionID("gmeans.critical", "Critical value for the Anderson Darling test. \u03B1=0.0001 is 1.8692, \u03B1=0.005 is 1.159 \u03B1=0.01 is 1.0348");

    /**
     * Stopping threshold
     */
    protected double delta;

    /**
     * Cluster model factory.
     */
    protected EMClusterModelFactory<O, M> mfactory;

    /**
     * Minimum number of iterations.
     */
    protected int miniter = 1;

    /**
     * Maximum number of iterations.
     */
    protected int maxiter = -1;

    /**
     * Prior to enable MAP estimation (use 0 for MLE)
     */
    double prior = 0.;

    /**
     * Retain soft assignments?
     */
    boolean soft = false;

    /**
     * Random number generator.
     */
    protected RandomFactory random;

    /**
     * critical value for AD-Test
     */
    protected double critical;

    @Override
    public void configure(Parameterization config) {
      new ObjectParameter<EMClusterModelFactory<O, M>>(MODEL_ID, EMClusterModelFactory.class, MultivariateGaussianModelFactory.class) //
          .grab(config, x -> mfactory = x);
      new DoubleParameter(DELTA_ID, 1e-7)// gui에서 디폴트값을 미리 주는것.
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE) //
          .grab(config, x -> delta = x);
      new IntParameter(MINITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> miniter = x);
      new IntParameter(MAXITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> maxiter = x);
      new DoubleParameter(PRIOR_ID) //
          .setOptional(true) //  gui에서 입력 안해도 넘어간다
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .grab(config, x -> prior = x);
      new Flag(SOFT_ID) //
          .grab(config, x -> soft = x);
      new RandomParameter(SEED_ID).grab(config, x -> random = x);
      new DoubleParameter(CRITICAL_ID, 1.8692) //
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .grab(config, x -> critical = x);
    }

    @Override
    public PGMeans_PCA_ADT make() {
      return new PGMeans_PCA_ADT(delta, mfactory, random, critical);
    }
  }
}
