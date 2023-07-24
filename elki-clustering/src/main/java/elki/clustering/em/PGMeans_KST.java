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


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.em.models.EMClusterModelFactory;
import elki.clustering.em.models.MultivariateGaussianModelFactory;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.projection.random.GaussianRandomProjectionFamily;
import elki.data.projection.random.RandomProjectionFamily.Projection;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDs;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.math.linearalgebra.CholeskyDecomposition;
import elki.math.linearalgebra.CovarianceMatrix;
import elki.math.statistics.distribution.NormalDistribution;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.*;
import elki.utilities.random.RandomFactory;
import static elki.math.linearalgebra.VMath.*;

import net.jafama.FastMath;

public class PGMeans_KST<O extends NumberVector, M extends MeanModel> implements ClusteringAlgorithm<Clustering<M>>{
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(PGMeans_KST.class);

  protected int k = 1;
  protected double delta;
  protected int p; // number of projections
  protected double alpha = 0.05; // significant level 0.05, dicuss: 프로젝트 추후에 알파에 따른 변화를 연구해봐도 좋다

  protected EMClusterModelFactory<? super O, M> mfactory;
  protected RandomFactory random;
  protected Random rand;

  /**
   *
   * Constructor.
   *
   * @param delta delta parameter
   * @param mfactory EM cluster model factory
   * @param p number of projections
   * @param random for Random Projection
   */
  public PGMeans_KST(double delta, EMClusterModelFactory<? super O, M> mfactory, int p, RandomFactory random){
    this.delta = delta;
    this.mfactory = mfactory;
    this.p = p;
    this.random = random;
    rand = this.random.getSingleThreadedRandom();
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
    EM<O, M> em = new EM<O, M>(k, delta, mfactory);
    while(rejected) {
      Clustering<M> clustering = em.run(relation);
      rejected = testResult(relation, clustering, p);
      if(rejected) {
        k++;
        System.out.println(k);
        em = new EM<O, M>(k, delta, mfactory);
      }
      // in general, the number of clusters is within 10
      if(k>100){
        System.out.println("KS-Test is going to be wrong");
        break;
      }
    }
    System.out.println("k :" + k);
    return em.run(relation);
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
  private boolean testResult(Relation<O> relation, Clustering<M> clustering, int p) {
    boolean rejected = false;

    for(int i=0; i<p; i++) {
      ArrayList<Cluster<M>> clusters = new ArrayList<>(clustering.getAllClusters());
      final int dim = RelationUtil.dimensionality(relation);

      for(Cluster<M> cluster : clusters) {
        if(cluster.size() < 1) continue;
        // TODO 제대로된 critical value구하는 식 찾기, 아래 4가지중에 
        // 1.
        // double dn = FastMath.sqrt(cluster.size()) - 0.01 + (0.83 / FastMath.sqrt(cluster.size())); 
        // double critical = 0.895 / dn; // n > 30
        // 2.
        // double critical = FastMath.sqrt(-.5 * FastMath.log(alpha/2)) / FastMath.sqrt(cluster.size()); // critical value for KS test Anpassungstest aus Miller, L.H.
        // 3.
        // double critical = FastMath.sqrt((3/alpha)/cluster.size()); // with sufficiently large n in literatur of PG-Means
        // 4.
        // double critical = 1.358 / FastMath.sqrt(cluster.size()); // if n > 35 and alpha = 0.05.
        // 5.
        double critical = generateCriticalValue(cluster.size());

        // generate random projection
        double[] P = generateMultivariateGaussianRandomProjection(dim);
        P = normalize(P);
        // TODO 여기서 애초에 데이터를 정규화 하고 그 정규화된 데이터로 mean과 cov를 만들까?
        NormalDistribution projectedNorm = projectedModel(cluster, relation, P);
        double[] projectedData = projectedData(cluster, relation, P);
        // then KS-Test with projected data and projected model
        double D = ksTest(projectedData, projectedNorm); // test statistic of KS-Test

        if(D > critical) {
          //rejected
          return rejected = true;
        }
      }
    }
    return rejected;
  }
  /**
   * generate the critical value for ks test
   * 
   * @param n the size of sample
   * @return critical value
   */
  // monte carlo simulation으로 n'= 3/alpha개의 D를 만들어서 그중에 95% quantile값을 구한다. 이렇게 구한값에 sqrt(n'/n)으로 스케일링을 하여 critical value를 구한다.
  private double generateCriticalValue(int n){
    double c = 0;
    // the number of Repeat count for simulation (n' = 3/alpha)
    int m = (int) FastMath.round(3/alpha);
    // TODO wie soll ich diese Situation umgehen?
    if(m >= n) {
      throw new IllegalArgumentException("not sufficiently large n");
    }
    double[] Dm = new double[m];
    // Monte Carlo Simulation
    for(int i=0; i<m; i++){
      double[] sample = new double[n];
      for(int j=0; j<n; j++){
        sample[j] = rand.nextGaussian(); // 랜덤 포인트 of 데이터
      }

      Dm[i] = ksTest(sample, new NormalDistribution(0, 1));
    }
    // choose the critical value (quantile(1-alpha))
    c = quantile(Dm, (1-alpha)*100 );
    // scaling the chosen critical value
    return c / (FastMath.sqrt(m) / FastMath.sqrt(n));
  }
  /**
   * compute the quantile 
   * 
   * @param data
   * @param percentile
   * @return qauntile of the @param data with @param percentile
   */
  private static double quantile(double[] data, double percentile) {
    Arrays.sort(data);
    int index = (int) Math.ceil(percentile / 100.0 * data.length) - 1;
    return data[index];
  }
  /**
   * project the data set
   *
   * @param cluster
   * @param relation
   * @param P projection
   * @return one dimensional projected data
   */
  private double[] projectedData(Cluster<? extends MeanModel> cluster, Relation<O> relation, double[] P) {
    DBIDs ids = cluster.getIDs();
    double[][] data = new double[ids.size()][];
    double[] projectedData = new double[ids.size()];
    // int dim = RelationUtil.dimensionality(relation);
    // double[] means = new double[dim];
    

    int i=0;
    // int count=0;
    // TODO do I have to standardize the data before the data are projected?
    // compute means
    for(DBIDIter iditer = ids.iter(); iditer.valid(); iditer.advance()) {
      O vec = relation.get(iditer);
      data[i++] = vec.toArray();
      // for(int j = 0; j < dim; j++) {
      //   means[j] += vec.doubleValue(j);
      // }
      // count++;
    }
    // Normalize mean
    // for(int j = 0; j < dim; j++) {
    //   means[j] /= count;
    // }
    // move the data to the center
    // for(int n=0; n < data.length; n++){
    //   for(int d=0; d < dim; d++){
    //     data[n][d] -= means[d];
    //   }
    // }
    for(int j=0; j<data.length; j++) {
      projectedData[j] = transposeTimes(P, data[j]);
    }
    return projectedData;
  }
  /**
   * project model
   *
   * @param cluster
   * @param relation
   * @param P projection
   * @return projected model
   */
  private NormalDistribution projectedModel(Cluster<? extends MeanModel> cluster, Relation<O> relation, double[] P) {
    CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster.getIDs());
    double[][] mat = cov.makePopulationMatrix();
    double projectedMean = transposeTimes(P, cov.getMeanVector());
    double projectedVar = transposeTimesTimes(P, mat, P);
    return new NormalDistribution(projectedMean, FastMath.sqrt(projectedVar));
  }
  
  /**
   * generate a multivariate gaussian random projection
   *
   * @param dim number of dimensions
   * @return multivariate gaussian random projection
   */
  private double[] generateMultivariateGaussianRandomProjection(int dim) {
    // create two array for Means and Covariance for random projection P, which is a matrix dim x 1
    double[] randomProjectionMeans = new double[dim];
    double[][] randomProjectionCov = new double[dim][dim];
    for(int i=0; i<dim; i++) {
      randomProjectionCov[i][i] = 1.0/dim;
    }

    CholeskyDecomposition chol = new CholeskyDecomposition(randomProjectionCov);
    double[][] L = chol.getL();
    double[] Z = generateRandomGaussian(L[0].length);

    return plus(times(L,Z), randomProjectionMeans);
  }
  // private double[] generateMultivariateGaussianRandomProjection(int dim) {
  //   double[] matrix = new double[dim];
  //   for(int i=0; i<matrix.length; i++){
  //     matrix[i] = rand.nextGaussian();
  //   }

  //   return matrix;
  // }
  /**
   * generate one dimensional random gaussian vector
   * @param n length of data
   * @return random gaussian vector
   */
  private double[] generateRandomGaussian(int n) {
    Random rand = random.getSingleThreadedRandom();
    double[] Z = new double[n];
    for(int i=0; i<n; i++) {
      Z[i] = rand.nextGaussian();
    }
    return Z;
  }

  /**
   * KS Test
   *
   * @param sample data
   * @param norm normal distribution
   * @return test statistic
   */
  private double ksTest(double[] sample, NormalDistribution norm) {
    int index = 0;
    double D = 0;

    Arrays.sort(sample);
    while(index < sample.length) {
      double x = sample[index];
      double model_cdf = norm.cdf(x);
      // Advance on first curve
      index++;
      // Handle multiple points with same x:
      while (index < sample.length && sample[index] == x) {
        index++;
      }
      double empirical_cdf = ((double) index + 1.) / (sample.length + 1.);
      D = Math.max(D, Math.abs(model_cdf - empirical_cdf));
    }
    return D;
  }

  // TODO TypeInformation이 뭐하는 역할인지 알기 (gui에서 입력을 해야하게끔 만들어주는것인가?)
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
     * Projection to specify the number of projections.
     */
    public static final OptionID NUMBER_OF_PROJECTIONS_ID = new OptionID("pgmeans.p", "Number of projections");

    /**
     * Randomization seed.
     */
    public static final OptionID SEED_ID = new OptionID("pgmeans.seed", "Random seed for splitting clusters.");

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
     * Number of projections
     */
    protected int p;

    /**
     * Random number generator.
     */
    protected RandomFactory random;


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
      new IntParameter(NUMBER_OF_PROJECTIONS_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
      		.grab(config, x -> p = x); //
      new RandomParameter(SEED_ID).grab(config, x -> random = x);
    }

    @Override
    public PGMeans_KST make() {
      return new PGMeans_KST(delta, mfactory, p, random);
    }
  }
}
