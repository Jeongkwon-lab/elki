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
import elki.database.ids.DBIDs;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.math.linearalgebra.CovarianceMatrix;
import elki.math.linearalgebra.pca.PCAResult;
import elki.math.linearalgebra.pca.PCARunner;
import elki.math.linearalgebra.pca.StandardCovarianceMatrixBuilder;
import elki.math.statistics.distribution.NormalDistribution;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.*;
import elki.utilities.random.RandomFactory;

import net.jafama.FastMath;

public class PGMeans_PCA_KST<O extends NumberVector, M extends MeanModel> implements ClusteringAlgorithm<Clustering<M>>{
    /**
    * Class logger
    */
    private static final Logging LOG = Logging.getLogger(PGMeans_PCA_KST.class);

    protected int k = 1;
    protected double delta;
    protected double alpha = 0.05; // significant level 0.05, dicuss: 프로젝트 추후에 알파에 따른 변화를 연구해봐도 좋다

    protected EMClusterModelFactory<? super O, M> mfactory;
    protected RandomFactory random;
    protected Random rand;
    protected double critical;

    /**
    *
    * Constructor.
    *
    * @param delta delta parameter
    * @param mfactory EM cluster model factory
    * @param mprojection Random projection family
    */
    public PGMeans_PCA_KST(double delta, EMClusterModelFactory<? super O, M> mfactory, RandomFactory random, double critical){
        this.delta = delta;
        this.mfactory = mfactory;
        this.random = random;
        rand = this.random.getSingleThreadedRandom();
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
        EM<O, M> em = new EM<O, M>(k, delta, mfactory);
        Clustering<M> clustering = em.run(relation);;
        while(rejected) {
          rejected = testResult(relation, clustering);
          if(rejected) {
            k++;
            System.out.println(k);
          }
          // in general, the number of clusters is within 100 for small data
          if(k>100){
            System.out.println("KS-Test is going to be wrong");
            break;
          }
          //TODO ?? repeat expectation-maximization 10times (maxiter=10) and then choose one result that has best Likelihood (that is EM-algorithm).
          em = new EM<O, M>(k, delta, mfactory, 10, false);
          clustering = em.run(relation);
        }
        
        System.out.println("k :" + k);
    
        return clustering;
    }

    /**
     * generate a random projection,
     * and project the dataset and model,
     * Then, KS-test
     *
     * @param relation
     * @param clustering the result of em with k
     * @return true if the test is rejected
     */
    private boolean testResult(Relation<O> relation, Clustering<M> clustering) {
        // pca depends on the variance of cluster. So we have not to repeat the test p-times
        ArrayList<Cluster<M>> clusters = new ArrayList<>(clustering.getAllClusters());

        double[][] projectedSamples = new double[clusters.size()][relation.size()];
        NormalDistribution[] projectedNorms = new NormalDistribution[clusters.size()];
        int j=0;

        // TODO soll ich mit jedem pca vector KS test ausprobieren?
        double[] pcaFilter = runPCA(relation);
        pcaFilter = normalize(pcaFilter);

        for(Cluster<M> cluster : clusters) {
            if(cluster.size() < 2) continue;
            // TODO 제대로된 critical value구하는 식 찾기
            // 0.886 / Math.sqrt(n) is from Lilliefors test table /// monte carlo -> lilliefors test table -> critical value
            //double critical = FastMath.sqrt(-.5 * FastMath.log(alpha/2)) / FastMath.sqrt(cluster.size()); // in wiki, it is Math.sqrt(-0.5 * Math.log(alpha/2)) / Math.sqrt(n)
            // double critical = Math.sqrt((3/alpha)/cluster.size()); // with sufficiently large n
            // double critical = generateCriticalValue(cluster.size());

            
            // double[][] data = new double[cluster.size()][RelationUtil.dimensionality(relation)];
            // int j=0;
            // for(DBIDIter iditer = cluster.getIDs().iter(); iditer.valid(); iditer.advance()) {
            //     O vec = relation.get(iditer);
            //     data[j++] = vec.toArray();
            // }
            projectedNorms[j] = projectedModel(relation, cluster, pcaFilter);
            projectedSamples[j] = projectedData(relation, cluster, pcaFilter);
            j++;
        }
        // then KS-Test with projected data and projected model
        // KSTest is too strict. It cannot be passed on large data.
        double D = ksTest(projectedSamples, projectedNorms); // test statistic of KS-Test

        if(D > critical) {
            //rejected
            return true;
        }

        return false;
    }
    /**
     * generate the critical value for ks test
     * 
     * @param n the size of sample
     * @return critical value
     */
    private double generateCriticalValue(int n){
        double c = 0;
        // the number of Repeat count for simulation (n' = 3/alpha)
        int m = (int) FastMath.round(3/alpha);
        if(m >= n) {
            throw new IllegalArgumentException("not sufficiently large n");
        }
        double[] Dm = new double[m];
        // Monte Carlo Simulation
        for(int i=0; i<m; i++){
        double[] sample = new double[n];
        for(int j=0; j<n; j++){
            sample[j] = rand.nextGaussian();
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
     * run PCA using ELKI
     *
     * @param relation
     * @param cluster 
     * @return filter that outputs one dimension matrix
     */
    private double[] runPCA(Relation<O> relation) {
        StandardCovarianceMatrixBuilder scov = new StandardCovarianceMatrixBuilder();
        PCARunner pca = new PCARunner(scov);
        PCAResult pcaResult = pca.processIds(relation.getDBIDs(), relation);

        // return first Vector in sorted EigenPairs, because the filter outputs one dimension
        double[][] eigenvectors = pcaResult.getEigenvectors();
        double[] filter = new double[eigenvectors.length];
        for(int i=0; i<filter.length; i++) {
            filter[i] = eigenvectors[i][0];
        }
        return filter;
    }
    /**
     * project the data set
     *
     * @param cluster
     * @param relation
     * @param P projection
     * @return one dimensional projected data
     */
    private double[] projectedData(Relation<O> relation, Cluster<? extends MeanModel> cluster, double[] P) {
        DBIDs ids = cluster.getIDs();
        double[][] data = new double[ids.size()][];
        double[] projectedData = new double[ids.size()];
        // CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster.getIDs());
        // double[] mean = cov.getMeanVector();
        // int dim = RelationUtil.dimensionality(relation);

        int i=0;
        // TODO do I have to standardize the data before the data are projected?
        for(DBIDIter iditer = ids.iter(); iditer.valid(); iditer.advance()) {
            O vec = relation.get(iditer);
            data[i++] = vec.toArray();
        }
        // move the data to the center : 여기서 하면 model projection할때도 해줘야한다
        // for(int n=0; n < data.length; n++){
        //     for(int d=0; d < dim; d++){
        //     data[n][d] -= mean[d];
        //     }
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
    private NormalDistribution projectedModel(Relation<O> relation, Cluster<? extends MeanModel> cluster, double[] P) {
        CovarianceMatrix cov = CovarianceMatrix.make(relation, cluster.getIDs());
        double[][] mat = cov.makePopulationMatrix();
        double projectedMean = transposeTimes(P, cov.getMeanVector());
        double projectedVar = transposeTimesTimes(P, mat, P);
        return new NormalDistribution(projectedMean, FastMath.sqrt(projectedVar));
    }

    /**
     * KS Test
     *
     * @param sample not sorted data
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
   * KS-tests on the reduced data and models in one dimension. 
   * 
   * @param sample sample data reduced to one dimension, stored per cluster in a one-dimensional array.
   * @param norm normal distribution for the reduced models that are stored per cluster in array.
   * @return maximum value of Dn
   */
  private double ksTest(double[][] sample, NormalDistribution[] norm) {
    double D = 0;

    for(int i=0; i<sample.length; i++){
      // if cluster.size() < 1, to avoid the null point exception.
      if(sample[i] == null || norm[i] == null) continue;

      int index = 0;
      Arrays.sort(sample[i]);
      while(index < sample[i].length) {
        double x = sample[i][index];
        double model_cdf = norm[i].cdf(x); 
        // Advance on first curve
        index++;
        // Handle multiple points with same x:
        while (index < sample[i].length && sample[i][index] == x) {
          index++;
        }
        double empirical_cdf = ((double) index) / (sample[i].length);
        D = Math.max(D, Math.abs(model_cdf - empirical_cdf));
      }
    }
    
    return D;
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
        public static final OptionID CRITICAL_ID = new OptionID("pgmeans.critical", "Critical value for the Kolmogorov Smirnov test.");


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
         * Critical value
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
        new DoubleParameter(CRITICAL_ID) //
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .grab(config, x -> critical = x);
        }

        @Override
        public PGMeans_PCA_KST make() {
            return new PGMeans_PCA_KST(delta, mfactory, random, critical);
        }
    }
}
