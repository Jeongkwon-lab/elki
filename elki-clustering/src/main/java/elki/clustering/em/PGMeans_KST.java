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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.em.models.EMClusterModel;
import elki.clustering.em.models.EMClusterModelFactory;
import elki.clustering.em.models.MultivariateGaussianModelFactory;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.EMModel;
import elki.data.model.MeanModel;
import elki.data.type.SimpleTypeInformation;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDataStore;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDs;
import elki.database.ids.ModifiableDBIDs;
import elki.database.relation.MaterializedRelation;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.logging.statistics.DoubleStatistic;
import elki.logging.statistics.LongStatistic;
import elki.math.statistics.distribution.NormalDistribution;
import elki.result.Metadata;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.Flag;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;
import net.jafama.FastMath;

public class PGMeans_KST<O extends NumberVector, M extends EMModel> implements ClusteringAlgorithm<Clustering<M>>{
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(PGMeans_KST.class);
  /**
   * Number of clusters
   */
  protected int k = 1;
  /**
   * Delta parameter
   */
  protected double delta;
  /**
   * Number of projections
   */
  protected int p;
  /**
   * Significant level
   */
  protected double alpha;
  /**
   * Factory for producing the initial cluster model.
   */
  protected EMClusterModelFactory<? super O, M> mfactory;
  /**
   * Factory for producing the random class
   */
  protected RandomFactory random;
  /**
   * Random class for producing the random Gaussian projection
   */
  protected Random rand;
  /**
   * Weight for each cluster
   */
  protected double[] w;
  /**
   * Best likelihood with 25 iterations of the EM algorithm 
   */
  protected double bestlikelihood = Double.NEGATIVE_INFINITY;
  /**
   * Clustering results for best likelihood
   */
  protected Clustering<M> bestClustering;
  /**
   * Iteration for the best likelihood
   */
  protected final int ITER = 25; // iteration number to find the bestlikelihood


  /**
   *
   * Constructor.
   *
   * @param delta delta parameter
   * @param mfactory EM cluster model factory
   * @param p number of projections
   * @param random for Random Projection
   * @param alpha confidence for ks test
   */
  public PGMeans_KST(double delta, EMClusterModelFactory<? super O, M> mfactory, int p, RandomFactory random, double alpha) {
    this.delta = delta;
    this.mfactory = mfactory;
    this.p = p;
    this.random = random;
    rand = this.random.getSingleThreadedRandom();
    this.alpha = alpha;
  }

  // TODO to compute the critical value instead of input of the value direct

  /**
   * Performs the PG-Means algorithm on the given database.
   *
   * @param relation relation
   * @return clustering result for PG-means
   */
  public Clustering<M> run(Relation<O> relation) {
    if(relation.size() == 0) {
      throw new IllegalArgumentException("database empty: must contain elements");
    }
    
    // PG-Means
    boolean rejected = true;
    Clustering<M> clustering = em(relation, -1, 1, false, 0.);
    while(rejected) {
      rejected = testResult(relation, clustering, p);
      if(rejected) {
        k++;
        System.out.println("number of k: " +k);
        // init weights
        this.w = new double[k];
        bestlikelihood = Double.NEGATIVE_INFINITY;
        //repeat expectation-maximization 25 times and then choose one result that has best Likelihood.
        for(int i=0; i<ITER; i++){
          em(relation, -1, 1, false, 0.);
        }
        clustering = bestClustering;
      }
      // in general, the number of clusters is within 100 for small data
      if(k>100){
        System.out.println("KS-Test is going to be wrong");
        break;
      }
    }
    
    System.out.println("k :" + k);

    return clustering;
  }

  /**
   * generate a random projection,
   * and project the dataset and model,
   * Then, run Kolmogorov Smirnov test
   *
   * @param relation relation
   * @param clustering the result of em with k
   * @param p number of projections
   * @return true if the test is rejected
   */
  private boolean testResult(Relation<O> relation, Clustering<M> clustering, int p) {
    ArrayList<Cluster<M>> clusters = new ArrayList<>(clustering.getAllClusters());
    double critical = lillie_cv(relation.size(), alpha);

    for(int i=0; i<p; i++) {
      final int dim = RelationUtil.dimensionality(relation);
      // generate random projection
      double[] P = generateGaussianRandomProjection(dim);
      // normalize the length of the projection 
      P = normalize(P);
      // project the data set
      double[] projectedSamples = projectedData(relation, P);
      NormalDistribution[] projectedModels = new NormalDistribution[clusters.size()];
      int j=0;
      for(Cluster<M> cluster : clusters){
        if(cluster.size() < 1) {
          j++;
          continue; 
        }
        // project the models
        projectedModels[j] = projectedModel(relation, cluster, P);
        j++;
      }
      double D = ksTest(projectedSamples, projectedModels);
      // double c = simulate_ks_cv(alpha, projectedModels, relation.size());
      if(D > critical) {
        //rejected
        return true;
      }
      
    }
    return false;
  }

  /**
   * Lilliefors critical value
   * 
   * @param n length of data
   * @param alpha significant level
   * @return critical value
   */
  private double lillie_cv(int n, double alpha){
    double dmax = 1.0;
    double deltaDmax = 0.5;
    if(n <= 100){
      for(int i=0; i<40; i++){
        double a = FastMath.exp(-7.01256 * FastMath.pow2(dmax) * (n + 2.78019) + 2.99587 * dmax * FastMath.sqrt(n + 2.78019) - .122119 + 0.974598 / FastMath.sqrt(n) + 1.67997/n);
        // double diff = a - alpha;
        if(a > alpha) dmax += deltaDmax;
        else dmax -= deltaDmax;
        deltaDmax /= 2;
      }
    }
    else {
      for(int i=0; i<40; i++){
        double a = FastMath.exp(-7.01256 * FastMath.pow2(dmax*FastMath.pow(n/100, 0.49)) * (100 + 2.78019) + 2.99587 * (dmax*FastMath.pow(n/100, 0.49)) * FastMath.sqrt(100 + 2.78019) - .122119 + 0.974598/FastMath.sqrt(100) + 1.67997/100);
        // double diff = a - alpha;
        if(a > alpha) dmax += deltaDmax;
        else dmax -= deltaDmax;
        deltaDmax /= 2;
      }
    }
    return dmax;
  }

  /**
   * generate for the critical value
   * 
   * @param alpha confidence
   * @param norms models
   * @param n length of the data
   */
  private double simulate_ks_cv(double alpha, NormalDistribution[] norms, int n){
    int numTrials = (int) (3 * FastMath.ceil(1/alpha));
    int simulation_n = k*20;
    if(n < simulation_n) simulation_n = n;
    double[] ksStats = new double[numTrials];
    // TODO n개의 랜덤 데이터가지고 numTrials만큼 ks test 반복. 
    // Monte Carlo Simulation
    for(int i=0; i<numTrials; i++){
      double[] sample = new double[simulation_n];
      for(int j=0; j<simulation_n; j++){
        // NormalDistribution rndNorm = norms[rand.nextInt(norms.length)];
        sample[j] = rand.nextGaussian();// * rndNorm.getStddev() + rndNorm.getMean();
      }
      double sampleMean = sum(sample) / sample.length;
      double sampleStd = FastMath.sqrt(squareSum(minus(sample, sampleMean)) / (sample.length-1));
      
      ksStats[i] = ksTest(sample, new NormalDistribution(sampleMean, sampleStd));
    }
    double cv = quantile(ksStats, (1-alpha)*100);
    return cv * FastMath.sqrt(simulation_n) / FastMath.sqrt(n);
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
   * EM clustering from ELKI
   * 
   * @param relation relation
   * @param maxiter Maximum number of iterations
   * @param miniter Minimum number of iterations
   * @param soft Include soft assignments
   * @param prior MAP prior
   * @return result for EM clustering
   */
  private Clustering<M> em(Relation<O> relation, int maxiter, int miniter, boolean soft, double prior) {
    if(relation.size() == 0) {
      throw new IllegalArgumentException("database empty: must contain elements");
    }
    final SimpleTypeInformation<double[]> SOFT_TYPE = new SimpleTypeInformation<>(double[].class);
    final String KEY = EM.class.getName();

    // initial models
    List<? extends EMClusterModel<? super O, M>> models = mfactory.buildInitialModels(relation, k);
    WritableDataStore<double[]> probClusterIGivenX = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_HOT | DataStoreFactory.HINT_SORTED, double[].class);
    double loglikelihood = EM.assignProbabilitiesToInstances(relation, models, probClusterIGivenX, null);
    DoubleStatistic likestat = new DoubleStatistic(this.getClass().getName() + ".loglikelihood");
    LOG.statistics(likestat.setDouble(loglikelihood));

    // iteration unless no change
    int it = 0, lastimprovement = 0;
    double bestloglikelihood = Double.NEGATIVE_INFINITY;// loglikelihood; // For
                                                        // detecting
                                                        // instabilities.
    for(++it; it < maxiter || maxiter < 0; it++) {
      final double oldloglikelihood = loglikelihood;
      EM.recomputeCovarianceMatrices(relation, probClusterIGivenX, models, prior);
      // reassign probabilities
      loglikelihood = EM.assignProbabilitiesToInstances(relation, models, probClusterIGivenX, null);

      LOG.statistics(likestat.setDouble(loglikelihood));
      if(loglikelihood - bestloglikelihood > delta) {
        lastimprovement = it;
        bestloglikelihood = loglikelihood;
      }
      if(it >= miniter && (Math.abs(loglikelihood - oldloglikelihood) <= delta || lastimprovement < it >> 1)) {
        break;
      }
    }
    LOG.statistics(new LongStatistic(KEY + ".iterations", it));

    // fill result with clusters and models
    List<ModifiableDBIDs> hardClusters = new ArrayList<>(k);
    for(int i = 0; i < k; i++) {
      hardClusters.add(DBIDUtil.newArray());
    }

    // provide a hard clustering
    for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
      hardClusters.get(argmax(probClusterIGivenX.get(iditer))).add(iditer);
    }
    Clustering<M> result = new Clustering<>();
    Metadata.of(result).setLongName("EM Clustering");
    // provide models within the result and assign the weights for each cluster
    double[] w_temp = new double[k];
    for(int i = 0; i < k; i++) {
      result.addToplevelCluster(new Cluster<>(hardClusters.get(i), models.get(i).finalizeCluster()));
      w_temp[i] = models.get(i).getWeight();
    }
    if(soft) {
      Metadata.hierarchyOf(result).addChild(new MaterializedRelation<>("EM Cluster Probabilities", SOFT_TYPE, relation.getDBIDs(), probClusterIGivenX));
    }
    else {
      probClusterIGivenX.destroy();
    }
    // update best likelihood, store the clustering result and assign the weights for the best likelihood
    if(this.bestlikelihood < bestloglikelihood){
      this.bestlikelihood = bestloglikelihood;
      this.bestClustering = result;
      w = w_temp.clone();
    }

    return result;
  }
  
  /**
   * project the data that is in @param cluster
   * 
   * @param relation relation
   * @param P is a projection that the data can be projected through
   * @return projected data of the data in @param cluster through @param P projection
   */
  private double[] projectedData(Relation<O> relation, double[] P) {
    DBIDs ids = relation.getDBIDs();
    double[][] data = new double[ids.size()][];
    double[] projectedData = new double[ids.size()];

    int i=0;
    for(DBIDIter iditer = ids.iter(); iditer.valid(); iditer.advance()) {
      O vec = relation.get(iditer);
      data[i++] = vec.toArray();
    }
    for(int j=0; j<data.length; j++) {
      projectedData[j] = transposeTimes(P, data[j]);
    }
    return projectedData;
  }
  /**
   * project the model of the data for @param cluster
   * 
   * @param relation relation
   * @param cluster cluster
   * @param P projection
   * @return projected model through projection @param P 
   */
  private NormalDistribution projectedModel(Relation<O> relation, Cluster<? extends M> cluster, double[] P) {
    EMModel emModel = cluster.getModel();
    double projectedMean = transposeTimes(P, emModel.getMean());
    double projectedVar = transposeTimesTimes(P, emModel.getCovarianceMatrix(), P);
    return new NormalDistribution(projectedMean, FastMath.sqrt(projectedVar));
  }
  
  /**
   * generate a gaussian random projection that reduces the dimensions to one dimension
   *
   * @param dim number of dimensions for input data
   * @return gaussian random projection
   */
  private double[] generateGaussianRandomProjection(int dim) {
    double[] projection = new double[dim];
    for(int i=0; i<projection.length; i++){
      projection[i] = rand.nextGaussian();
    }
    return projection;
  }

  /**
     * Kolmogorov Smirnov Test for one sample
     *
     * @param sample not sorted data sample
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

  /**
   * Kolmogorov Smirnov Test on the reduced data and models 
   *
   * @param sample data samples reduced to one dimension
   * @param norm normal distributions for the reduced models that are stored per cluster in array
   * @return test statistic
   */
  private double ksTest(double[] sample, NormalDistribution[] norm) {
    int index = 0;
    double D = 0;

    Arrays.sort(sample);
    while(index < sample.length) {
      double x = sample[index];
      double model_cdf = cdfForMixtureModel(x, norm);
      // Advance on first curve
      index++;
      // Handle multiple points with same x:
      while (index < sample.length && sample[index] == x) {
        index++;
      }
      double empirical_cdf = ((double) index) / (sample.length);
      D = Math.max(D, Math.abs(model_cdf - empirical_cdf));
    }
    return D;
  }

  /**
   * cumulative distribution function for the Gaussian mixture model
   * 
   * @param x value
   * @param norm normal distributions reduced to one dimension
   * @return cdf value for @param x
   */
  private double cdfForMixtureModel(double x, NormalDistribution[] norm){
    double cdf = 0;
    for(int i=0; i<norm.length; i++){
      if(norm[i] == null) continue;
      cdf += w[i] * norm[i].cdf(x);
    }
    return cdf;
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
     * Projection to specify the number of projections.
     */
    public static final OptionID NUMBER_OF_PROJECTIONS_ID = new OptionID("pgmeans.p", "Number of projections");

    /**
     * Randomization seed.
     */
    public static final OptionID SEED_ID = new OptionID("pgmeans.seed", "Random seed for splitting clusters.");

    /**
     * Critical value for the Anderson-Darling-Test
     */
    public static final OptionID CRITICAL_ID = new OptionID("pgmeans.critical", "Critical value for the Kolmogorov Smirnov test.");

    /**
     * Critical value for the Anderson-Darling-Test
     */
    public static final OptionID ALPHA_ID = new OptionID("pgmeans.alpha", "Confidence value for the Kolmogorov Smirnov test.");

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

    /**
     * confidence alpha
     */
    protected double alpha;


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
      new DoubleParameter(ALPHA_ID) //
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .grab(config, x -> alpha = x);
    }

    @Override
    public PGMeans_KST make() {
      return new PGMeans_KST(delta, mfactory, p, random, alpha);
    }
  }
}
