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

import java.util.Arrays;
import java.util.List;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.em.models.EMClusterModel;
import elki.clustering.em.models.EMClusterModelFactory;
import elki.clustering.em.models.MultivariateGaussianModelFactory;
import elki.clustering.kmeans.quality.ApproximateWeightOfEvidence;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.minkowski.EuclideanDistance;
import elki.logging.Logging;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.Flag;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import static elki.clustering.kmeans.quality.AbstractKMeansQualityMeasure.*;

import net.jafama.FastMath;

import static elki.clustering.em.EM.*;
import static elki.math.linearalgebra.VMath.*;

public class BayesianApproach <O, M extends MeanModel, V extends NumberVector> implements ClusteringAlgorithm<Clustering<M>> {
  /**
   * Class logger
   */
  private static final Logging LOG = Logging.getLogger(PGMeans_KST.class);
  
  protected int k = 1;
  protected double delta;
  protected int max_k;
  protected EMClusterModelFactory<? super O, M> mfactory;
  protected double[] AWE;
  /**
   * Constructor.
   *
   * @param delta delta parameter
   * @param mfactory EM cluster model factory
   */
  public BayesianApproach(double delta, EMClusterModelFactory<? super O, M> mfactory, int max_k) {
    this.mfactory = mfactory;
    this.delta = delta;
    this.max_k = max_k;
    this.AWE = new double[this.max_k];
  }
  /**
   * Performs the algorithm using Approximate Bayesian Approach on the given database.
   * 
   * @param relation
   * @return result
   */
  public Clustering<M> run(Relation<O> relation){
    if(relation.size() == 0) {
      throw new IllegalArgumentException("database empty: must contain elements");
    }
    // approximate Weight of Evidence 
    double[] AWE_k = new double[max_k];
    ApproximateWeightOfEvidence awe = new ApproximateWeightOfEvidence();
    while(k<=max_k) {
      EM<O, M> em_k = new EM<O, M>(k, delta, mfactory);
//      EM<O, M> em_k1 = new EM<O, M>(k+1, delta, mfactory);
      Clustering<M> clustering_k = em_k.run(relation);
//      Clustering<M> clustering_k1 = em_k1.run(relation);
      AWE_k[k-1] = awe.quality(clustering_k, EuclideanDistance.STATIC, (Relation<V>) relation);
      //AWE_k[k-1] = quality(clustering_k, clustering_k1, EuclideanDistance.STATIC, (Relation<V>) relation);
      k++;
    }
    // AWE값 클러스터 갯수별로 출력
    for(int i=0; i<max_k; i++) {
      System.out.print(AWE_k[i] + ", ");
    }
    System.out.println("");
    // AWE를 비교후 최고의 AWE를 갖는 클러스터의 갯수 k를 골라 EM알고리즘을 돌려 결과를 출력
    k = argmax(AWE_k) + 1;
    EM<O, M> em = new EM<O, M>(k, delta, mfactory);
    return em.run(relation);
  }
  
//  private <V extends NumberVector> double quality(Clustering<? extends MeanModel> clustering1, Clustering<? extends MeanModel> clustering2, NumberVectorDistance<? super V> distance, Relation<V> relation) {
//    double LL1 = logLikelihood(relation, clustering1, distance); // Null hypothese
//    double LL2 = logLikelihood(relation, clustering2, distance); // alternative hypothese
//    double lambda_LR = 2 * (LL2 - LL1);
//    int df = numberOfFreeParameters(relation, clustering2) - numberOfFreeParameters(relation, clustering1);
//    int n = numPoints(clustering1);
//    
//    return  lambda_LR - (2*df * (1.5 + FastMath.log(n)));
//  }
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
     * Parameter to specify the saving of soft assignments
     */
    public static final OptionID MAX_K_ID = new OptionID("awe.max_k", "Maximum number of clusters."); // gui에 나타날 표시 및 설명

    /**
     * Stopping threshold
     */
    protected double delta;

    /**
     * Cluster model factory.
     */
    protected EMClusterModelFactory<O, M> mfactory;
    
    /**
     * Maximum number of clusters
     */
    protected int max_k;

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
      new IntParameter(MAX_K_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .grab(config, x -> max_k = x);
    }

    @Override
    public BayesianApproach make() {
      return new BayesianApproach(delta, mfactory, max_k);
    }
  }
}
